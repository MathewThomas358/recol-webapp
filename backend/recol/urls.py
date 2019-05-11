from django.urls import path
from . import views


app_name = "recol"

urlpatterns = [
    path("", views.homepage, name="homepage"),
    path("classify", views.classify, name="classify")
]
