
#from django.urls import path
from diagnosis.urls import path
from diagnosis.views import upload_file
from . import views

urlpatterns = [
   # path('/diagnosis/views/upload/', upload_file, name='upload_file'),
   # from home page go to any page
    path('', views.home, name='home'),
    path('diagnosis/', views.Diagnosis, name='Diagnosis'),
    path('services/', views.services, name='services'),
    path('care/', views.care, name='care'),
    path('Research/', views.Research, name='Research'),
    # from diagnosis page go to any page
     
]


