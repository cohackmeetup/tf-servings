import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model = tf.keras.models.load_model('./models/keras_inception.h5')
export_path = './production_models/prod_inception/1'

with tf.keras.backend.get_session() as sess: # We always need to start a tf session to do anything
    tf.saved_model.simple_save( # this is the command to save a file in .pb format
        sess, 
        export_path,
        inputs={'input_image': model.input},  # here we need to specify the input, model.input will be a Tensor
        outputs={t.name: t for t in model.outputs}) # mapping the outputs
