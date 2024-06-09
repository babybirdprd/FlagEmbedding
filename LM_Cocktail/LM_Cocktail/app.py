from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utils
from LM_Cocktail import mix_models, mix_models_with_data

app = FastAPI()

# Define request models for each endpoint
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

# Mix models endpoint
@app.post("/mix_models")
async def mix_models_endpoint(request: MixModelsRequest):
    try:
        model = cocktail.mix_models(**request.dict())
        if request.output_path:
            model.save_pretrained(request.output_path)
        return {"message": "Models mixed successfully", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mix models with data endpoint
@app.post("/mix_models_with_data")
async def mix_models_with_data_endpoint(request: MixModelsWithDataRequest):
    try:
        model = cocktail.mix_models_with_data(**request.dict())
        if request.output_path:
            model.save_pretrained(request.output_path)
        return {"message": "Models mixed with data successfully", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mix models by layers endpoint
@app.post("/mix_models_by_layers")
async def mix_models_by_layers_endpoint(request: MixModelsByLayersRequest):
    try:
        model = cocktail.mix_models_by_layers(**request.dict())
        if request.output_path:
            model.save_pretrained(request.output_path)
        return {"message": "Models mixed by layers successfully", "output_path": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
