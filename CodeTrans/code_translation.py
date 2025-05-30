# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from comps import MegaServiceEndpoint, MicroService, ServiceOrchestrator, ServiceRoleType, ServiceType
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from fastapi import Request
from fastapi.responses import StreamingResponse

MEGA_SERVICE_PORT = int(os.getenv("MEGA_SERVICE_PORT", 7777))
LLM_SERVICE_HOST_IP = os.getenv("LLM_SERVICE_HOST_IP", "0.0.0.0")
LLM_SERVICE_PORT = int(os.getenv("LLM_SERVICE_PORT", 9000))
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


def align_inputs(self, inputs, cur_node, runtime_graph, llm_parameters_dict, **kwargs):
    """Aligns the inputs based on the service type of the current node.

    Parameters:
    - self: Reference to the current instance of the class.
    - inputs: Dictionary containing the inputs for the current node.
    - cur_node: The current node in the service orchestrator.
    - runtime_graph: The runtime graph of the service orchestrator.
    - llm_parameters_dict: Dictionary containing the LLM parameters.
    - kwargs: Additional keyword arguments.

    Returns:
    - inputs: The aligned inputs for the current node.
    """

    # Check if the current service type is EMBEDDING
    if self.services[cur_node].service_type == ServiceType.EMBEDDING:
        # Store the input query for later use
        self.input_query = inputs["query"]
        # Set the input for the embedding service
        inputs["input"] = inputs["query"]

    # Check if the current service type is RETRIEVER
    elif self.services[cur_node].service_type == ServiceType.RETRIEVER:
        # Extract the embedding from the inputs
        embedding = inputs["data"][0]["embedding"]
        # Align the inputs for the retriever service
        inputs = {"index_name": llm_parameters_dict["index_name"], "text": self.input_query, "embedding": embedding}
    elif self.services[cur_node].service_type == ServiceType.LLM:
        # convert TGI/vLLM to unified OpenAI /v1/chat/completions format
        next_inputs = {}
        next_inputs["model"] = LLM_MODEL_ID
        next_inputs["messages"] = [{"role": "user", "content": inputs["query"]}]
        next_inputs["max_tokens"] = llm_parameters_dict["max_tokens"]
        next_inputs["top_p"] = llm_parameters_dict["top_p"]
        next_inputs["stream"] = inputs["stream"]
        next_inputs["frequency_penalty"] = inputs["frequency_penalty"]
        next_inputs["temperature"] = inputs["temperature"]
        inputs = next_inputs

    return inputs

class CodeTransService:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        ServiceOrchestrator.align_inputs = align_inputs
        self.megaservice = ServiceOrchestrator()
        self.endpoint = str(MegaServiceEndpoint.CODE_TRANS)

    def add_remote_service(self):
        llm = MicroService(
            name="llm",
            host=LLM_SERVICE_HOST_IP,
            port=LLM_SERVICE_PORT,
            api_key=OPENAI_API_KEY,
            endpoint="/v1/chat/completions",
            use_remote_service=True,
            service_type=ServiceType.LLM,
        )
        self.megaservice.add(llm)

    async def handle_request(self, request: Request):
        data = await request.json()
        language_from = data["language_from"]
        language_to = data["language_to"]
        source_code = data["source_code"]
        prompt_template = """
            ### System: Please translate the following {language_from} codes into {language_to} codes.

            ### Original codes:
            '''{language_from}

            {source_code}

            '''

            ### Translated codes:
        """
        prompt = prompt_template.format(language_from=language_from, language_to=language_to, source_code=source_code)
        result_dict, runtime_graph = await self.megaservice.schedule(initial_inputs={"query": prompt})
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="codetrans", choices=choices, usage=usage)

    def start(self):
        self.service = MicroService(
            self.__class__.__name__,
            service_role=ServiceRoleType.MEGASERVICE,
            host=self.host,
            port=self.port,
            endpoint=self.endpoint,
            input_datatype=ChatCompletionRequest,
            output_datatype=ChatCompletionResponse,
        )
        self.service.add_route(self.endpoint, self.handle_request, methods=["POST"])
        self.service.start()


if __name__ == "__main__":
    service_ochestrator = CodeTransService(port=MEGA_SERVICE_PORT)
    service_ochestrator.add_remote_service()
    service_ochestrator.start()
