import tensorflow as tf
from tensorflow import keras
import google.cloud.storage as storage
from tensorflow.keras.utils import load_img

def classify_image(event, context):

  print(event)
  
  # Get the file that was just uploaded
  bucket_name = event['bucket']
  file_name = event['name']

  # Load the pre-trained model
  model_bucket_name = 'my-thesis-models'
  model_file_name = 'model_VGG16_COVID.h5'
  client = storage.Client()
  model_bucket = client.get_bucket(model_bucket_name)
  model_blob = model_bucket.blob(model_file_name)
  model_blob.download_to_filename('/tmp/model.h5')
  model = tf.keras.models.load_model('/tmp/model.h5')

  # Load the image and make a prediction
  image_bucket = client.get_bucket(bucket_name)
  image_blob = image_bucket.blob(file_name)
  image_blob.download_to_filename('/tmp/image.jpg')
  image = load_img('/tmp/image.jpg', target_size=(224,224))
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = image/255
  image = tf.expand_dims(image, axis=0)
  prediction = model.predict(image)
  prediction2 = model.predict(image)[0][0] * 100

  # Save the prediction to the result bucket
  result_bucket_name = 'thesis-prediction-result'
  result_file_name = file_name.replace('.jpg', '.txt')
  result_bucket = client.get_bucket(result_bucket_name)
  result_blob = result_bucket.blob(result_file_name)
  result_blob.upload_from_string(str(prediction))
  result_blob.upload_from_string('The patient has {:.2f}% chance of Covid-19.'.format(prediction2))
