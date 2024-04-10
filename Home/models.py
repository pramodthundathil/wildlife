from django.db import models

# Create your models here.

class Animal(models.Model):
    animal = models.CharField(max_length=255)
    date = models.DateTimeField(auto_now_add=True)
