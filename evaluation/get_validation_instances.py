from datasets import load_dataset
import json

def dataset_to_json(dataset_name, split, save_path):
    # Load the dataset from Huggingface
    dataset = load_dataset(dataset_name)

    json_dataset = []
    for example in dataset[split]:
        example['FAIL_TO_PASS'] = json.loads(example['FAIL_TO_PASS'])
        example['PASS_TO_PASS'] = json.loads(example['PASS_TO_PASS'])
        json_dataset.append(example)
    # Write to a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(json_dataset, json_file, indent=4)

def print_image_name(dataset_name, split):
    dataset = load_dataset(dataset_name)
    for example in dataset[split]:
        print(f"{example['instance_id']}\t\t\tzzr/{example['repo'].replace('/', '__')}__{example['version']}")

# Call the function
# Note: The 'princeton-nlp/SWE-bench_Lite' dataset may not have the key 'validation', so it might need to be adjusted.
dataset_name = 'princeton-nlp/SWE-bench_Lite'

# Save to a JSON file (for example in the current working directory)
# save_json_path = 'dataset/swebench_lite_dev.json'
# dataset_to_json(dataset_name, 'dev', save_json_path)
# save_json_path = 'dataset/swebench_lite_test.json'
# dataset_to_json(dataset_name, 'test', save_json_path)
# print(f'Dataset saved to {save_json_path}')

print_image_name(dataset_name, 'dev')