import profile
from pyexpat import model
from statistics import mode
from django.db import models
from account.models import Profile

# Create your models here.
class UploadImage(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    image = models.ImageField(upload_to = 'uploads')
    result = models.CharField(max_length=100, null=True, blank= True)
    date = models.DateField(auto_now_add=True)