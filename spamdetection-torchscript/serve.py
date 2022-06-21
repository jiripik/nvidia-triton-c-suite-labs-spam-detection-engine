import torch
import json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def model_fn(model_dir):
    return torch.jit.load('model.pt').eval().to(device)

def predict_fn(input_data, model):
    model.to(device)
    model.eval()

    input_id, attention_mask = input_data
    input_id = torch.LongTensor([input_id])
    attention_mask = torch.LongTensor([attention_mask])

    input_id = input_id.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        return model(input_id, attention_mask)[0]       

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)  
    return data['input_ids'], data['attention_mask']
