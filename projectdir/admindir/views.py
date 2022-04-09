from importlib.resources import contents
from django.contrib import messages
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import HttpResponseForbidden
import numpy as np
import os
import sys
from scipy import ndimage
import re
import imageio
from pathlib import Path
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):
    if(request.session.has_key('account_id')):
        content = {}
        content['title'] = 'Admin'
        return render(request, 'admin/index.html', content)
    else:
        return HttpResponseRedirect(reverse('account-login'))