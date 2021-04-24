import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from IPython import display
import time

(train_X, train_y), (_, _) = cifar10.load_data()

num_epcohs = 500
lr_discrim = 0.0002
beta_1_discrim = 0.5 
lr_gen = 0.0002
beta_1_gen = 0.5
batch_size = 100
z_dim = 100
CONTINUE_TRAINING = True
start_point = 0

# load the parameter
(train_X, _), (_, _) = cifar10.load_data()

# the data configuration
img_size_cifar = 32
num_channels_cifar = 3 
img_size_flat_cifar = img_size_cifar * img_size_cifar*num_channels_cifar
img_shape_cifar = (img_size_cifar, img_size_cifar, num_channels_cifar)
num_classes_cifar = 10


# process the data
train_X = train_X/255
train_X = train_X*2-1
train_Y = to_categorical(train_y, num_classes_cifar)

kernel_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)


# define the discriminator
def discriminator(input_images, reuse=False):
    
    with tf.variable_scope('discriminator', reuse= reuse):
        

        
        layer1 = tf.layers.conv2d(input_images, filters=64, 
                                  kernel_size=3, strides=2, 
                                  padding='same', kernel_initializer=kernel_init, name='conv1')

        layer1 = tf.nn.leaky_relu(layer1, alpha=0.2, name='leaky_relu1')
    
        
        layer2 = tf.layers.conv2d(layer1, 
                                  filters=128, 
                                  kernel_size=3, 
                                  strides=2, 
                                  padding='same', 
                                  kernel_initializer=kernel_init, 
                                  name='conv2')
        
        layer2 = tf.nn.leaky_relu(layer2, alpha=0.2, name='leaky_relu2')

        layer3 = tf.layers.conv2d(layer2, 
                                  filters=128, 
                                  kernel_size=3, 
                                  strides=2, 
                                  padding='same', 
                                  kernel_initializer=kernel_init, 
                                  name='conv3')
        
        layer3 = tf.nn.leaky_relu(layer3, alpha=0.2, name='leaky_relu3')
 


        layer4 = tf.layers.conv2d(layer3, 
                                 filters=256, 
                                 kernel_size=3, 
                                 strides=2,
                                 padding='same',
                                 name='conv4')
        # layer3 = tf.layers.batch_normalization(layer3,momentum=0.99, training=is_training, name='batch_normalization3')
        layer4 = tf.nn.leaky_relu(layer4, alpha=0.2, name='leaky_relu4')
        
        
        layer4 = tf.layers.flatten(layer4)
        # layer4 = tf.reshape(layer4, (-1, layer4.shape[1]*layer4.shape[2]*layer4.shape[3]))
        layer4 = tf.nn.dropout(layer4, keep_prob=0.6)
        
        logits_discrim= tf.layers.dense(layer4, 1)
        
        output_discrim = tf.sigmoid(logits_discrim)
        
        
        # build the aux layer
        net_1 = tf.layers.dense(inputs=layer4, name='aux_fc1', units=128, activation=tf.nn.relu)
        logits_label = tf.layers.dense(inputs=net_1, name='aux_fc_out', units=num_classes_cifar, activation=None)
        output_label_one_hot = tf.nn.softmax(logits=logits_label)
        output_label_s = tf.argmax(output_label_one_hot, dimension=1)
        
        return logits_discrim, logits_label, output_label_s
    
    
# define generator 

def generator(z, fake_label, reuse=False):
    
    with tf.variable_scope('generator', reuse=reuse):
        

        input_to_conv = tf.layers.dense(tf.concat([z, fake_label], axis=1), 4*4*512)
        

        layer1 = tf.reshape(input_to_conv, (-1, 4, 4, 512))
        layer1 = tf.nn.leaky_relu(layer1, alpha=0.2, name='leaky_relu1_g')
        
        
        layer2 = tf.layers.conv2d_transpose(layer1, filters=256, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution2')
        layer2 = tf.nn.leaky_relu(layer2, alpha=0.2, name='leaky_relu2_g')
        
 
        layer3 = tf.layers.conv2d_transpose(layer2, filters=128, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution3')
        
        
        layer3 = tf.nn.leaky_relu(layer3, alpha=0.2, name='leaky_relu3_g')
        

        layer4 = tf.layers.conv2d_transpose(layer3, filters=256, kernel_size=5, strides= 2, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution4')
        
        layer4 = tf.nn.leaky_relu(layer4, alpha=0.2, name='leaky_relu4_g')
        
        

        layer5 = tf.layers.conv2d_transpose(layer4, filters=3, kernel_size=5, strides=1, padding='same', 
                                            kernel_initializer=kernel_init, name='deconvolution5')
           
        
        logits = tf.tanh(layer5, name='tanh')
        
        return logits
    
    
x = tf.placeholder(tf.float32, shape= (None, img_size_cifar, img_size_cifar, num_channels_cifar), name="d_input")
label_true = tf.placeholder(tf.float32, [None, num_classes_cifar], name="label_true")

z = tf.placeholder(tf.float32, shape= (None, z_dim), name="z_noise")
label_fake = tf.placeholder(tf.float32, shape= (None, num_classes_cifar), name="label_fake")

is_training = tf.placeholder(tf.bool, [], name='is_training')



fake_x=generator(z, label_fake)

D_logit_real_discrim, D_logit_real_label, D_output_real_label = discriminator(x, reuse=False)

D_logit_fake_discrim, D_logit_fake_label, D_output_real_label = discriminator(fake_x, reuse=True)



D_loss_real_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real_discrim),
                                                                     logits=D_logit_real_discrim))

D_loss_real_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_true,
                                                                     logits=D_logit_real_label))

D_loss_fake_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake_discrim),
                                                                     logits=D_logit_fake_discrim))

D_loss_fake_label = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_fake,
                                                                     logits=D_logit_fake_label))


D_loss = D_loss_real_discrim + D_loss_real_label + D_loss_fake_discrim + D_loss_fake_label

G_loss_discrim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake_discrim),
                                                                logits=D_logit_fake_discrim))

G_loss = G_loss_discrim + D_loss_fake_label

training_vars = tf.trainable_variables()

theta_D = [var for var in training_vars if var.name.startswith('discriminator')]
theta_G = [var for var in training_vars if var.name.startswith('generator')]

d_optimizer = tf.train.AdamOptimizer(lr_discrim, beta_1_discrim).minimize(D_loss, var_list=theta_D)
g_optimizer = tf.train.AdamOptimizer(lr_gen, beta_1_gen).minimize(G_loss, var_list=theta_G)



saver = tf.train.Saver()

def get_fake_labels(num_labels):
    fake_label_value = np.random.randint(num_classes_cifar, size=num_labels)
    fake_label_value = fake_label_value.reshape(-1,1)
    # print(fake_label_value)
    fake_label_value = to_categorical(fake_label_value, num_classes_cifar)
    return fake_label_value

num_batches = int(train_X.shape[0] / batch_size)
loss_tracker_epoch = []



with tf.Session() as sess_1:
    
    #initialize all variables
    sess_1.run(tf.global_variables_initializer())
    
    #for each epcohs
    for epoch in range(num_epcohs):
        
        
        total_index = np.arange(len(train_X))
        
        np.random.shuffle(total_index)
        train_X_1 = np.take(a=train_X, indices=total_index, axis=0)
        train_Y_1 = np.take(a=train_Y, indices=total_index, axis=0)
        
        discrim_loss_list = []
        gen_loss_list = []
        
        #for number of batches
        for i in range(num_batches):
            
            start_time = time.time()
            #select start and end of the batch
            start = i * batch_size
            end = (i + 1) * batch_size
            
            #sample batch images
            batch_images = train_X_1[start:end]
            batch_label_real = train_Y_1[start:end]
            batch_label_fake = get_fake_labels(batch_size)
            z_noise = np.random.uniform(-1,1, size=[batch_size, z_dim]).astype(np.float32)

            
            #train the discriminator after every two steps
            if(i % 2 == 0):
                
                #train the discriminator
                _, discrim_loss = sess_1.run([d_optimizer,D_loss], feed_dict={x: batch_images, label_true:batch_label_real,
                                                                              label_fake:batch_label_fake, z: z_noise})

               
            
            #train the generator and discriminator
            _, gen_loss = sess_1.run([g_optimizer,G_loss], feed_dict={x: batch_images,
                                                                      label_fake:batch_label_fake, z: z_noise})
            
            
            _, discrim_loss = sess_1.run([d_optimizer,D_loss], feed_dict={x: batch_images, label_true:batch_label_real,
                                                                          label_fake: batch_label_fake, z: z_noise})
        
            
            print("Epoch: {}, Iter: {},  Discrim_loss:{}, Gen_loss: {}, it takes: {}s".format(epoch, i, discrim_loss, gen_loss, time.time()-start_time))
            discrim_loss_list.append(discrim_loss)
            gen_loss_list.append(gen_loss)
            
        loss_tracker_epoch.append([np.mean(discrim_loss_list), np.mean(gen_loss_list)])
        np.save("loss_tracker.npy", loss_tracker_epoch)
        saver.save(sess_1, "./saved_models/model_backup.ckpt")
        
        if (epoch+1) % 5 == 0 and epoch > 0:
            saver.save(sess_1, "./saved_models/model_{}.ckpt".format(epoch))