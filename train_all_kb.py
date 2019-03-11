import config
import models
import os

MODELS = ['analogy',
          'complex',
          'distmult',
          'hole',
          'rescal',
          'transd',
          'transe',
          'transh',
          'transr']

os.environ['CUDA_VISIBLE_DEVICES']='-1'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./benchmarks/KB/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

MODELS = ['analogy',
          'complex',
          'distmult',
          'hole',
          'rescal',
          'transd',
          'transe',
          'transh',
          'transr']

for model in MODELS:

    if model == 'analogy':
        model_obj = models.Analogy
    elif model == 'complex':
        model_obj = models.ComplEx
    elif model == 'distmult':
        model_obj = models.DistMult
    elif model == 'hole':
        model_obj = models.HolE
    elif model == 'rescal':
        model_obj = models.RESCAL
    elif model == 'transd':
        model_obj = models.TransD
    elif model == 'transe':
        model_obj = models.TransE
    elif model == 'transh':
        model_obj = models.TransH
    elif model == 'transr':
        model_obj = models.TransR
    else:
        raise FileNotFoundError("Can't find {} in models/".format(model))

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files("./res/{}.vec.tf".format(model), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files("./res/{}_embedding.vec.json".format(model))
    # Initialize experimental settings.
    con.init()
    # Set the knowledge embedding model
    con.set_model(model_obj)
    # Train the model.
    con.run()
    # To test models after training needs "set_test_flag(True)".
    # con.test()
    # con.predict_head_entity(152, 9, 5)
    # con.predict_tail_entity(151, 9, 5)
    # con.predict_relation(151, 152, 5)
    # con.predict_triple(151, 152, 9)
    # con.predict_triple(151, 152, 8)
