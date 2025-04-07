from AVH_sam_model import load_sam_model
from AVH_load_image import load_and_convert_BGR2RGB_image
from AVH_mask_predictor import predictor_mask,predict_masks_method
from AVH_mask_maker import detect_edges_detail,draw_contours_mask_smooth_or_raw
import base64
import cv2
from fastapi import FastAPI, UploadFile, File,Query,HTTPException, Body
from fastapi.responses import FileResponse
import io
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
import signal
import shutil
import subprocess
from typing import List,Optional
import uvicorn


from pathlib import Path
import os

app = FastAPI()
mask_predictor, mask_generator = load_sam_model()
BASE_IMAGE_PATH = "/home/appuser/Grounded-Segment-Anything/AVH/images"
# Vérifier si le dossier existe déjà
if not os.path.exists(BASE_IMAGE_PATH):
    os.makedirs(BASE_IMAGE_PATH)


sessions = {}


####test###########
@app.post("/stop-server/")
async def stop_server():
    try:
        os.kill(os.getpid(), signal.SIGINT)
        return {"message": "Server is shutting down."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping/")
async def root():
    return {"message": "pong"}

################

class ImageSession:
    def __init__(self):
        self.image_path = None  # Chemin de l'image
        self.mask = None        # Masque associé à l'image
        self.name = None        # Nom de l'image (optionnel)
        self.last_file_number = None # Stocke le dernier numéro de fichier utilisé

    def upload_file(self, file, image_id):
        # Chemin où les images seront sauvegardées
        save_path = Path(BASE_IMAGE_PATH)

        # Chemin du dossier spécifique à l'image
        image_folder = save_path / image_id

        # Vérification de l'existence du dossier, création s'il n'existe pas
        if not image_folder.exists():
            os.makedirs(image_folder)

        # Chemin complet de l'image
        self.image_path = image_folder / file.filename

        # Sauvegarde de l'image dans le dossier spécifique
        with open(self.image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Chargement et conversion de l'image pour la prédiction
        image = load_and_convert_BGR2RGB_image(str(self.image_path))

        # Prédiction du masque pour l'image
        predictor_mask(mask_predictor, image)

        # Retourne le chemin complet de l'image
        return "upload image succès"
    
    def predict(self, image_id, input_point=None, input_label=None, input_box=None):
        input_point = np.array(input_point) if input_point else None
        input_label = np.array(input_label) if input_label else None
        input_box = np.array(input_box) if input_box else None

        masks, score = predict_masks_method(mask_predictor, input_point=input_point, input_label=input_label,
                                            input_box=input_box, multimask_output=False)
        self.mask = masks
        self.name = image_id

        mask_folder = Path(BASE_IMAGE_PATH) / image_id / "mask"
        if not mask_folder.exists():
            os.makedirs(mask_folder)

        # Recherche du prochain numéro de fichier disponible dans le dossier mask
        existing_files = [int(file.stem) for file in mask_folder.glob("*.png") if file.stem.isdigit()]
        if existing_files:
            # next_number = max(existing_files) + 1
            next_number = max(existing_files)
        else:
            next_number = 1
        self.last_file_number = next_number
        file_name = f"{next_number}.png"
        file_path = mask_folder / file_name

        # Charger l'image à partir du masque
        image_to_show = Image.fromarray(masks[0])
        image_to_show.save(file_path)
        
        return file_path
    
    def maskmaker(self,image_id, edges, smooth, line_smoother, R, G, B, 
              line_width, threshold1, threshold2, kernel_width, kernel_height, 
              L2gradient, dilation_enabled, apertureSize, sigma, ksizes):



        color = (R, G, B)
        ksize = (ksizes, ksizes)
        
        if not self.image_path:
            return "No image uploaded for this session."


        edges_folder = Path(BASE_IMAGE_PATH) / image_id / "edges"
        if not edges_folder.exists():
            os.makedirs(edges_folder)

        if edges:
            mask_detect_edges = detect_edges_detail(str(self.image_path), self.mask, index=0, ksize=ksize, sigma=sigma, threshold1=threshold1,
                                                    threshold2=threshold2, apertureSize=apertureSize, L2gradient=L2gradient,
                                                    dilation_enabled=dilation_enabled, kernel_width=kernel_width, kernel_height=kernel_height, color=color)
            
            # Recherche du prochain numéro de fichier disponible dans le dossier mask
            if self.last_file_number is None:
                return {"error": "No previous file number available."}

            mask_edges = cv2.cvtColor(mask_detect_edges, cv2.COLOR_BGR2RGB)
            file_name = f"{self.last_file_number}.png"
            file_path = edges_folder / file_name

            # file_path = os.path.join(save_path, file_name)
            cv2.imwrite(str(file_path), mask_edges)
       
        else:
            mask_draw_contours = draw_contours_mask_smooth_or_raw(self.mask, index=0, smooth=smooth, line_smoother=line_smoother, color=color, line_width=line_width)
            
             # Recherche du prochain numéro de fichier disponible dans le dossier mask
            if self.last_file_number is None:
                return {"error": "No previous file number available."}

            file_name = f"{self.last_file_number}.png"
            file_path = edges_folder / file_name
            
            
            image_to_show = Image.fromarray(mask_draw_contours)
            # save_path = "/home/appuser/Grounded-Segment-Anything/AVH"
            # file_name = f"contour_.png"
            # file_path = os.path.join(save_path, file_name)
            image_to_show.save(file_path)
        
    
        return file_path


@app.post("/upload/{image_id}")
async def upload_file(image_id: str, file: UploadFile = File(...)):
    # Vérifie si une session existe pour cet identifiant d'image, sinon crée une nouvelle session
    if image_id not in sessions:
        sessions[image_id] = ImageSession()
   
    path = sessions[image_id].upload_file(file, image_id)

    # Retourne le chemin complet de l'image qui a été téléchargée
    return {"path": str(path)}

@app.get("/get_image/")
async def get_image(image_id: str = Query(...), image_name: str = Query(...)):
    image_path = Path(f"{BASE_IMAGE_PATH}/{image_id}/{image_name}")
    if not image_path.is_file():
        return {"error": f"Image '{image_name}' not found for ID '{image_id}' on the server"}
    return FileResponse(image_path)


@app.post("/predict/{image_id}")
async def predict(image_id: str, input_point: Optional[List[List[int]]] = Body(None), 
                  input_label: Optional[List[int]] = Body(None), input_box: Optional[List[int]] = Body(None)):
    if image_id not in sessions:
        return {"error": "Image session not found."}
    file_path = sessions[image_id].predict(image_id, input_point, input_label, input_box)

    # with open(file_path, "rb") as f:
    #     data = f.read()
    #     encoded_data = base64.b64encode(data).decode("utf-8")

    # # Renvoyer l'image encodée en base64 dans la réponse
    # return {"image_base64": encoded_data}
    return FileResponse(file_path)


@app.post("/maskmaker/{image_id}")
async def maskmaker(image_id: str, 
                    edges: Optional[bool] = Query(False),
                    smooth: Optional[bool] = Query(False), 
                    line_smoother: Optional[float] = Query(0.001),
                    R: Optional[int] = Query(0), G: Optional[int] = Query(0), 
                    B: Optional[int] = Query(0), line_width: Optional[int] = Query(3),
                    threshold1: Optional[int] = Query(1), 
                    threshold2: Optional[int] = Query(1),
                    kernel_width: Optional[int] = Query(2),
                    kernel_height: Optional[int] = Query(2),
                    L2gradient: Optional[bool] = Query(True),
                    dilation_enabled: Optional[bool] = Query(True), 
                    apertureSize: Optional[int] = Query(3),
                    sigma: Optional[int] = Query(1),
                    ksizes: Optional[int] = Query(0)):

    if image_id not in sessions:
        return {"error": "Image session not found."}
    file_path = sessions[image_id].maskmaker(image_id, edges, smooth, line_smoother, R, G, B, 
                                             line_width, threshold1, threshold2, 
                                             kernel_width, kernel_height, L2gradient, 
                                             dilation_enabled, apertureSize, sigma, ksizes)
    # with open(file_path, "rb") as f:
    #     data = f.read()
    #     encoded_data = base64.b64encode(data).decode("utf-8")

    # # Renvoyer l'image encodée en base64 dans la réponse
    # return {"image_base64": encoded_data}
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)