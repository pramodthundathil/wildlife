from django.shortcuts import render, redirect
from .detection import Analyse

def Index(request):
    return render(request,"index.html")

def CallCam(request):
    Analyse()
    return redirect(Index)
