from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from .recol import recol_fn
from .cnn_function import cnn
from django.contrib import messages

import cv2
import os

@csrf_exempt
def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]
    valid_extensions = ['.jpg', '.png', '.jpeg']
    if not ext.lower() in valid_extensions:
        ret = "Invalid Image. Please select a valid image."
    else:
        ret = ""

    return ret

def homepage(request):
    if request.method == "POST":
        image_ul = request.FILES['test_image']
        ret = validate_file_extension(image_ul)
        if not ret:
            image_ul.name = 'input.png'
            fs = FileSystemStorage()
            fs.save(image_ul.name, image_ul)
        else:
            messages.success(request, f"{ret}")
            #os.remove("media/input.png")

    return render(request, 'recol/upload.html')

def classify(request):
    if request.method == "POST":
        input = cv2.imread("media/input.png")
        kp = recol_fn(input)
        os.remove("media/input.png")
        pred = cnn(kp)
        messages.success(request, f"{pred}")

    return render(request, 'recol/classify.html')
