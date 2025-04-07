import cv2
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch



def show_box(box, ax):
    """
    Affiche une boîte sur une frame matplotlib.

    Args:
        box (numpy.ndarray): Les coordonnées de la boîte d'encadrement [x_min, y_min, x_max, y_max].
        ax (matplotlib.axes.Axes): Les axes matplotlib sur lesquels afficher la boîte.
    """
    # Extraire les coordonnées de la boîte
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    # Ajouter un rectangle sur les axes avec les coordonnées de la boîte
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    """
    Affiche les points avec un code couleur spécifiques dans un axe donné.

    Args:
        coords (numpy.ndarray)([X,Y]): Coordonnées des points.
        label (numpy.ndarray)([0] ou [1]): Étiquettes des points (1 pour les points positifs "garder", 0 pour les points négatifs"enlever").
        ax: Objet d'axe matplotlib où afficher les points.
        marker_size (int): Taille du marqueur pour l'affichage des points.

    """
    # Sélection des points positifs et négatifs
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    # Affichage des points positifs en vert avec un marqueur en forme d'étoile
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # Affichage des points négatifs en rouge avec un marqueur en forme d'étoile
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def show_mask(mask, ax, random_color=False):
    """
    Affiche un masque dans un axe donné.

    Args:
        mask (numpy.ndarray): Masque à afficher.
        ax: Objet d'axe matplotlib où afficher le masque.
        random_color (bool): Indique si une couleur aléatoire doit être utilisée pour le masque.

    Returns:
        None
    """
    # Génération d'une couleur aléatoire
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    # Récupération des dimensions du masqu
    h, w = mask.shape[-2:]
    # Création de l'image du masque en multipliant le masque par la couleur
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # Affichage du masque dans l'axe donné
    ax.imshow(mask_image)


def show_image(image,masks=None,index=None, input_point=None,input_label=None,input_box=None, mask=False, point=False,box=False):
    """
    Affiche une image dans une figure matplotlib.

    Args:
        image (numpy.ndarray): L'image à afficher.
        masks: Les masques associés à l'image.
        index (int): L'index du masks à afficher si le masks est une liste de masks.
        input_point (numpy.ndarray)([X,Y]): Les coordonnées des points d'entrée.
        input_label (numpy.ndarray)([0] ou [1]): Étiquettes des points (1 pour les points positifs "garder", 0 pour les points négatifs"enlever").
        input_box(numpy.ndarray)([x_min, y_min, x_max, y_max]): Boîte d'entrée pour la prédiction.
        mask (bool): Indique si l'image est un masque ou non.
        point (bool): Indique si les points doivent être affichés ou non.
        box (bool):Indique si la box doivent être affichés ou non.
    """
    plt.figure(figsize=(8, 8))
    # Affichage du mask
    if mask:
        image_to_show = Image.fromarray(image[index])
        plt.imshow(image_to_show, cmap='gray')
    # Affichage de la box   
    elif box :
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
    # Affichage de l'image
    else:
        plt.imshow(image)

    # Affichage des points
    if point:
        #plt.gca().imshow(image)
        show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()


def show_mask_score(image,masks,scores,input_point,input_label):
    """
    Affiche les masques et les scores associés sur une image donnée.

    Args:
        image (numpy.ndarray): L'image sur laquelle afficher les masques.
        masks (list(numpy.ndarray)): Liste des masques à afficher.
        scores (list(float)): Liste des scores associés aux masques.
         input_point (numpy.ndarray)([X,Y]): Les coordonnées des points d'entrée.
        input_label (numpy.ndarray)([0] ou [1]): Étiquettes des points (1 pour les points positifs "garder", 0 pour les points négatifs"enlever").
    """

    # Affichage des masques et des scores
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        # Affiche le masque sur l'image
        show_mask(mask, plt.gca())
        # Affiche les points d'entrée 
        show_points(input_point, input_label, plt.gca())  
        plt.title(f"Masque {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        #mask = masks.astype(np.uint8) * 255