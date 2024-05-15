from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://192.168.100.210:6700/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="/hdd2/zzr/models/llama3-instruct-70b/",
                                      prompt="San Francisco is a")
print("Completion result:", completion)