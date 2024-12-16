import os

import torch
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from transformers import T5EncoderModel, T5Tokenizer

# ==========================================================
# os.environ['CUDA_CACHE_DISABLE'] = '0' # orig is 0
# os.environ['CUDA_FORCE_PTX_JIT'] = '1'# no orig
# os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

mixed_precision.set_global_policy('mixed_float16')

# ==========================================================
PER_RESIDUE = True
PER_PROTEIN = False
SEC_STRUCT = False
MCAPST5_CHECKPOINT = "mcapst5_pan_epoch_20.hdf5"
XGBOOST_CHECKPOINT = "xgboost_pan_epoch_20.bin"

BATCH_SIZE = 64
SEQ_SIZE = 1200
DIM = 1024
# ==========================================================

# Load T5 model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))
t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
t5_model = t5_model.to(device) # move model to GPU
t5_model = t5_model.eval() # set model to evaluation model
t5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

# Load checkpoint
model = tf.keras.models.load_model(MCAPST5_CHECKPOINT)
model_ = XGBClassifier()
model_.load_model(XGBOOST_CHECKPOINT)

def get_embeddings(seqs, max_residues=4000, max_seq_len=1000, max_batch=100):
    # if SEC_STRUCT:
    #   sec_struct_model = load_sec_struct_model()

    results = {"residue_embs" : dict(),
               "protein_embs" : dict(),
               "sec_structs" : dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = t5_tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = t5_model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            # if sec_struct: # in case you want to predict secondary structure from embeddings
            #   d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                # if sec_struct: # get classification results
                #     results["sec_structs"][identifier] = torch.max( d3_Yhat[batch_idx,:s_len], dim=1 )[1].detach().cpu().numpy().squeeze()
                if PER_RESIDUE: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                # if PER_PROTEIN: # apply average-pooling to derive per-protein embeddings (1024-d)
                #     protein_emb = emb.mean(dim=0)
                #     results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    return results["residue_embs"]

def func(i, embedding_dict, pair_dataframe):
    i = i.numpy() # Decoding from the EagerTensor object
    x1= pad(embedding_dict[pair_dataframe['p1'][i]])
    x2= pad(embedding_dict[pair_dataframe['p2'][i]])
    y = pair_dataframe['label'][i]
    return x1, x2, y

def _fixup_shape(x1, x2, y):
    x1.set_shape((SEQ_SIZE, DIM))
    x2.set_shape((SEQ_SIZE, DIM))
    y.set_shape(())

    return (x1, x2), y

def pad(rst, length=1200, dim=1024):
    if len(rst) > length:
        return rst[:length]
    elif len(rst) < length:
        return np.concatenate((rst, np.zeros((length - len(rst), dim))))
    return rst

def predict(seq_a: str, seq_b: str):
    # Load example fasta.
    testing_seqs = {
        "A": seq_a,
        "B": seq_b,
    }

    for id, seq in testing_seqs.items():
        if len(seq) > 1200:
            testing_seqs[id] = seq[:1200]

    # Compute embeddings and/or secondary structure predictions
    embedding_dict = get_embeddings(testing_seqs)
    print(len(embedding_dict["A"]))
    print(len(embedding_dict["B"]))
    print(len(embedding_dict))

    pair_dataframe = pd.DataFrame({
        "p1": ["A"],
        "p2": ["B"],
        "label": [0],
    })

    test_dataset = tf.data.Dataset.from_generator(lambda: range(len(pair_dataframe)), tf.uint64).map(
        lambda i: tf.py_function(func=func,
                                 inp=[i, embedding_dict, pair_dataframe],
                                 Tout=[tf.float16,
                                       tf.float16, tf.float16]
                                 ),
        num_parallel_calls=tf.data.AUTOTUNE).map(_fixup_shape).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(model.layers[-2].name).output)

    # Use intermediate layer to transform pairs matrix
    pred = intermediate_layer_model.predict(test_dataset)
    y_pred = model_.predict(pred)

    return y_pred

print("----------")
print(predict("MIIIRYLVRETLKSQLAILFILLLIFFCQKLVKILGAAVDGEIPTNLVLSLLGLGIPEMAQLILPLSLFLGLLMTLGKLYTESEITVMHACGLSKAVLVKAAMILALFTGIVAAVNVMWAGPMSSRHQDEVLAEAKANPGMAALAQGQFQQATDGNSVLFIESVDGSKFNDVFLAQLRTKGNARPSVVVADSGQLAQRKDGSQVVTLNKGTRFEGTAMLRDFRITDFQNYQAIIGDPTDTEQMDMRTLWNTDTDRARAEFHWRITLVFTVFMMALIVVPLSVVNPRQGRVLSMLPAMLLYLIYFLLQTSIRSNGAKGKLDPMVWTWFVNSLYILLALGLNLWDTVPVRRI",
              "GVLDRYIGKTIFTTIMMTLFMLVSLSGIIKFVDQLKKAGQGSYDALGAGMYTLLSVPKDVQIFFPMAALLGALLGLGMLAQRSELVVMQASGFTRLQVALSVMKTAIPLVLLTMAIGEWVAPQGEQMARNYRAQAMYGGSLLSTQQGLWAKDGQNFVYIERVKGDDELGGVSIYAFNDERRLQSVRHASSAKFDPEHKQWRLSQVDESDLTNPKQITGSQTVSGTWKTNLTPDKLGVVALDPDALSISGLHNYVKYLKSSGQDAGRYQLNMWSKIFQPMSVAVMMLMALSFIFGPLRSVPMGVRVVTGISFGFVFYVLDQIFGPLTLVYGIPPIIGALLPSASFLLISLWLLLKR"))