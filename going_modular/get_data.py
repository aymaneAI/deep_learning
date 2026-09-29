from pathlib import Path
import os, zipfile
import requests
# Create directors
def download_data(file_name:str,
                  data_url:str,
                  ):
                    
  '''
  file_name : the name of the file using for storing the data
  data_url : the url of the data you want to download
  '''
                    
  data_path = Path('data/')
  image_path = data_path/file_name
  image_path_zip = data_path/Path(data_url).name
  if image_path.is_dir():
    print(f'{image_path} already exist ,skip the download.')
  else:
    print(f'{image_path} does not exist. Create one...')
    image_path.mkdir(parents = True, exist_ok = True)

    with open(image_path_zip, 'wb') as f:
      print('Download the file...')
      request = requests.get(data_url)
      f.write(request.content)
    with zipfile.ZipFile(image_path_zip, 'r') as zip_ref:
      print('Unzip file data.')
      zip_ref.extractall(image_path)

    os.remove(image_path_zip)
  train_dir = image_path/'train'
  test_dir  = image_path/'test'
  return image_path, train_dir, test_dir
