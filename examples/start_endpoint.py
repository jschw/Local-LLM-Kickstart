from llm_kickstart.llm_kickstart import LLMKickstart

manager = LLMKickstart()
# Start a new endpoint
manager.create_endpoint("Local_LLM_Model")
# List all running endpoints
manager.list_processes()
