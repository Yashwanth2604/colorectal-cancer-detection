from django.urls import path

from . import views

urlpatterns = [path('', views.index, name="index"),
               path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('LoadDataset', views.LoadDataset, name="LoadDataset"),
	       path('DetectCancer', views.DetectCancer, name="DetectCancer"),
	       path('DetectCancerAction', views.DetectCancerAction, name="DetectCancerAction"),	
	       path('Aboutus', views.Aboutus, name="Aboutus"),
	       path('BookAppointmentAction', views.BookAppointmentAction, name="BookAppointmentAction"),	

	       path('RunInception', views.RunInception, name="RunInception"),
	       path('RunResnet', views.RunResnet, name="RunResnet"),
	       path('RunEfficientNet', views.RunEfficientNet, name="RunEfficientNet"),
]