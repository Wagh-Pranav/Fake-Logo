from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from joblib import load
from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
import re

# Create your views here.
def index(request):
    if(request.session.has_key('account_id')):
        content = {}
        content['title'] = 'Home'
        return render(request, 'home/index.html', content)
    else:
        return HttpResponseRedirect(reverse('account-login'))
