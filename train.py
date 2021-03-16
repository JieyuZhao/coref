#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import util
import logging
format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  config = util.initialize_from_env()
  
  model = util.get_model(config)
  saver = tf.train.Saver()
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]


  log_dir = config["log_dir"]
  max_steps = config['num_epochs'] * config['num_docs']
  print("--max_steps:", max_steps)

  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  mode = 'w'


  with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state("data/spanbert_base")
    if ckpt and ckpt.model_checkpoint_path:
      checkpoint_path = os.path.join("data/spanbert_base", "model.max.ckpt")
      print("Restoring from: {}".format(checkpoint_path))
      saver.restore(session, checkpoint_path)
      mode = 'a'
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    trigger_token_ids = [[170, 170, 170]]
    cnt = 0
    while True:
      if cnt != 0:
        next_input = session.run(model.input_tensors, feed_dict={model.trigger_token_ids: trigger_token_ids})
        next_input[0][0][1:4] = trigger_token_ids[0]
        tf_loss, tf_global_step, _, batch_grad, input_tensors, = session.run([model.loss, model.global_step, \
          model.train_op, model.batch_grad, model.instance_tensors], \
            feed_dict={i:t for i,t in zip(model.input_tensors, next_input)})
      else:
        tf_loss, tf_global_step, _, batch_grad, input_tensors, = session.run([model.loss, model.global_step, \
          model.train_op, model.batch_grad, model.instance_tensors], \
            feed_dict={model.trigger_token_ids: trigger_token_ids})

      
      print('--tf global_step', tf_global_step)
      # print("outside, input_tensors:", input_tensors)
      # print("tf_loss:", tf_loss)
      
      
      trigger_token_ids = model.get_trigger_token_ids(session, input_tensors)
      
      # # trigger_token_ids = input_tensors[-1]
      # # cand_trigger_token_ids = session.run(model.hotflip_attack(batch_grad[0].values[1:4],
      # #                                                   model.word_embeddings,
      # #                                                   trigger_token_ids,
      # #                                                   num_candidates=40))

      # # trigger_token_ids = model.get_best_candidates(session,
      # #                                               input_tensors,
      # #                                                 trigger_token_ids,
      # #                                                 cand_trigger_token_ids,
      # #                                                 ) 
      # print("new trigger_token_ids:", trigger_token_ids)

      # print("==========="*4)

      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0
        logger.info(f"[{cnt}]-th triggers={', '.join([model.vocab[x] for x in trigger_token_ids[0]])}")

      cnt += 1
      if tf_global_step  > 0 and tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        print("current trigger token ids:", trigger_token_ids)
        eval_summary, eval_f1 = model.evaluate(session, tf_global_step, trigger_token_ids = trigger_token_ids)
        print(f"max_f1: {max_f1}; eval_f1:{eval_f1}")
        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        # if tf_global_step > max_steps:
        if cnt > max_steps:
          break
