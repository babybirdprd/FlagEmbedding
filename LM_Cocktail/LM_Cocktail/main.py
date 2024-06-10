from fastapi import FastAPI, HTTPException, Request
from typing import List, Union, Optional
from pydantic import BaseModel, Field
from LM_Cocktail import mix_models, mix_models_with_data, mix_models_by_layers

app = FastAPI()

# Define request models
class MixModelsRequest(BaseModel):
    models: Optional[List[str]] = Field(None, alias='model_names_or_paths')
    model_type: str
    weights: List[float]
    output_path: str

class ExampleDataItemForLLM(BaseModel):
    input: str
    output: str

class ExampleDataItemForEmbedder(BaseModel):
    query: str
    pos: List[str]
    neg: List[str]

class MixModelsWithDataRequest(BaseModel):
    models: Optional[List[str]] = Field(None, alias='model_names_or_paths')
    model_type: str
    example_data: Union[List[ExampleDataItemForLLM], List[ExampleDataItemForEmbedder]]
    temperature: float
    output_path: str
    max_input_length: Optional[int] = None
    neg_number: Optional[int] = None

class MixModelsByLayersRequest(BaseModel):
    models: Optional[List[str]] = Field(None, alias='model_names_or_paths')
    model_type: str
    weights: List[float]
    output_path: str

@app.post("/mix_models")
async def mix_models_endpoint(req: MixModelsRequest):
    try:
        model = mix_models(
            model_names_or_paths=req.models,
            model_type=req.model_type,
            weights=req.weights,
            output_path=req.output_path
        )
        return {"message": "Model mixed successfully.", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mix_models_with_data")
async def mix_models_with_data_endpoint(req: MixModelsWithDataRequest):
    try:
        if req.model_type == 'encoder':
            example_data = [{"query": item.query, "pos": item.pos, "neg": item.neg} for item in req.example_data]
            model = mix_models_with_data(
                model_names_or_paths=req.models,
                model_type=req.model_type,
                example_data=example_data,
                temperature=req.temperature,
                output_path=req.output_path,
                max_input_length=req.max_input_length,
                neg_number=req.neg_number
            )
        else:  # Assuming the default is decoder
            example_data = [{"input": item.input, "output": item.output} for item in req.example_data]
            model = mix_models_with_data(
                model_names_or_paths=req.models,
                model_type=req.model_type,
                example_data=example_data,
                temperature=req.temperature,
                output_path=req.output_path
            )
        return {"message": "Model mixed with data successfully.", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mix_models_by_layers")
async def mix_models_by_layers_endpoint(req: MixModelsByLayersRequest):
    try:
        model = mix_models_by_layers(
            model_names_or_paths=req.models,
            model_type=req.model_type,
            weights=req.weights,
            output_path=req.output_path
        )
        return {"message": "Model mixed by layers successfully.", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using: uvicorn <filename_without_py>:app --reload
