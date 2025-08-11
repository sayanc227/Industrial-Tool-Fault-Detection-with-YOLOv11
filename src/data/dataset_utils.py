
import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml, data_dir='/content/data'):
  """
  Creates a data.yaml configuration file for Ultralytics YOLO training.

  Args:
    path_to_classes_txt (str): Path to the classes.txt file containing class names.
    path_to_data_yaml (str): Path where the data.yaml file will be saved.
    data_dir (str): Root directory containing train and validation image/label folders.
  """
  # Read class.txt to get class names
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Create data dictionary
  data = {
      'path': data_dir,
      'train': 'train/images',
      'val': 'validation/images',
      'nc': number_of_classes,
      'names': classes
  }

  # Write data to YAML file
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Created config file at {path_to_data_yaml}')
