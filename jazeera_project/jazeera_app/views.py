from django.shortcuts import render

# Create your views here.
def isit(request):
    return render(request, 'base.html')