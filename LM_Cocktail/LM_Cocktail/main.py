import os
import shutil
import torch
import random
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models

from utils import load_model, get_model_param_list, merge_param, compute_weights, get_model_param_dirs, merge_param_by_layer

app = FastAPI()

class MixModelsRequest(BaseModel):
    model_names_or_paths: List[str]
    model_type: str
    weights: List[float]
    output_path: str = None

class MixModelsWithDataRequest(BaseModel):
    model_names_or_paths: List[str]
    model_type: str
    example_data: List[Dict]
    temperature: float = 5.0
    batch_size: int = 2
    max_input_length: int = 2048
    neg_number: int = 7
    output_path: str = None

class MixModelsByLayersRequest(BaseModel):
    model_names_or_paths: List[str]
    model_type: str
    weights: List[float]
    output_path: str = None

def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normalized: bool = True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normalized:
        normalized_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalized_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)

@app.post("/mix_models")
def mix_models_endpoint(request: MixModelsRequest):
    try:
        model = mix_models(request.model_names_or_paths, request.model_type, request.weights, request.output_path)
        return {"message": "Model mixed successfully", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mix_models_with_data")
def mix_models_with_data_endpoint(request: MixModelsWithDataRequest):
    try:
        model = mix_models_with_data(
            request.model_names_or_paths, request.model_type, request.example_data, 
            request.temperature, request.batch_size, request.max_input_length, 
            request.neg_number, request.output_path
        )
        return {"message": "Model mixed successfully with data", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mix_models_by_layers")
def mix_models_by_layers_endpoint(request: MixModelsByLayersRequest):
    try:
        model = mix_models_by_layers(request.model_names_or_paths, request.model_type, request.weights, request.output_path)
        return {"message": "Model mixed successfully by layers", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def mix_models(model_names_or_paths: List[str], model_type: str, weights: List[float], output_path: str = None):
    assert len(model_names_or_paths) == len(weights)
    assert model_type in ['decoder', 'encoder', 'reranker']
    assert sum(weights) - 1 <= 1e-3

    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)
    new_param = merge_param(param_list, weights=weights)

    model = load_model(model_names_or_paths[0], model_type=model_type)
    model.load_state_dict(new_param)

    if output_path is not None:
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

        if model_type == "encoder":
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model

def mix_models_with_data(model_names_or_paths: List[str], model_type: str, example_data: List[Dict], temperature: float = 5.0, batch_size: int = 2, max_input_length: int = 2048, neg_number: int = 7, output_path: str = None):
    assert model_type in ['decoder', 'encoder', 'encoder-decoder']

    model = load_model(model_names_or_paths[0], model_type=model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
    param_list = get_model_param_list(model_names_or_paths, model_type=model_type)

    weights = compute_weights(model, tokenizer=tokenizer, param_list=param_list, model_type=model_type, example_data=example_data, temperature=temperature, neg_number=neg_number, batch_size=batch_size, max_input_length=max_input_length)

    new_param = merge_param(param_list, weights=weights)
    model.load_state_dict(new_param)

    if output_path is not None:
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        if model_type == "encoder":
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model

def mix_models_by_layers(model_names_or_paths: List[str], model_type: str, weights: List[float], output_path: str = None):
    assert len(model_names_or_paths) == len(weights)
    assert model_type in ['decoder', 'encoder', 'reranker']
    assert sum(weights) - 1 <= 1e-3

    param_dirs, temp_dir = get_model_param_dirs(model_names_or_paths, model_type=model_type)
    temp_file_path = merge_param_by_layer(param_dirs, weights=weights)

    with init_empty_weights():
        if model_type == 'decoder':
            meta_model = AutoModelForCausalLM.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        elif model_type == 'encoder':
            meta_model = AutoModel.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        elif model_type == 'reranker':
            meta_model = AutoModelForSequenceClassification.from_pretrained(model_names_or_paths[0], trust_remote_code=True)
        else:
            raise NotImplementedError(f"not support this model_type: {model_type}")

    device_map = {name: "cpu" for name, _ in meta_model.named_modules()}
    model = load_checkpoint_and_dispatch(meta_model, checkpoint=temp_file_path, device_map=device_map)
    model.tie_weights()

    os.remove(temp_file_path)
    shutil.rmtree(temp_dir)

    if output_path is not None:
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(model_names_or_paths[0])
        tokenizer.save_pretrained(output_path)

        if model_type == "encoder":
            save_ckpt_for_sentence_transformers(ckpt_dir=output_path)
    return model
