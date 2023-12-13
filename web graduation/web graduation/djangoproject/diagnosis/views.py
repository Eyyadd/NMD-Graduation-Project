from django.shortcuts import render

def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['csv_file']
        with open('uploaded_file.csv', 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        return render(request, 'upload_success.html')
    return render(request, 'upload.html')
