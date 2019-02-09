import tensorflow as tf
def train_function(train_images,train_labels,val_images,val_labels, Regularization, Learn_rate, batch_size, layers):
    """Comment discription here later
    """
    
    # variables
    train_num_examples = train_images.shape[0]
    val_img_examples =  val_images.shape[0]

    #specify network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    layer_input_list=[x]
    with tf.name_scope('linear_model') as scope:
        for layer in range(0,layers):
            vars()['hidden_layer_'+str(layer+1)]=tf.layers.dense(layer_input_list[layer], 400, activation=tf.nn.relu,kernel_regularizer=Regularization,bias_regularizer=Regularization, name='hidden_layer_'+str(layer+1))
            layer_input_list.append(vars()['hidden_layer_'+str(layer+1)])
        output = tf.layers.dense(vars()['hidden_layer_'+str(layer+1)], 10, name='output_layer')
        tf.identity(output, name='output')    
                    
    #evaluation
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(output, 1)),"float"), name='accuracy')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output,name='ce_loss')

    # set up training
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = Learn_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()
    save_directory = './homework1_logs'

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        best_val_acc=0
        for epoch in range(100):
            print('Epoch',epoch)
            # run gradient steps
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
                session.run([train_op, tf.reduce_mean(cross_entropy)], {x: batch_xs, y: batch_ys})

            #train accuraccy
            train_acc =  session.run(accuracy, {x: train_images, y: train_labels})
            print('Training Accuracy',train_acc)

            #validation accuracy
            val_acc = session.run(accuracy, {x: val_images,y: val_labels})
            print('Validation Accuracy',val_acc)


            #early stopping
            if (val_acc > best_val_acc ):
                best_val_acc=val_acc
                best_train_acc=train_acc
                counter=0
            else:
                counter=counter+1
                if counter > 5:
                    break
                else:
                    continue
                    
    path_prefix = saver.save(session, os.path.join(save_directory, "homework_1"), global_step=global_step_tensor)
    tf.reset_default_graph()

    return best_train_acc, best_val_acc
