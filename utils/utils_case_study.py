
import numpy as np
import pandas as pd
import torch

from kgat.data_loader import tok2int_list_concat, FNNDataLoader, tok2int_list_case_study


def load_case_study(tokenizer, args):

    example_df = pd.read_csv(f"C:\\Workspace\\Rumor\\Demo\\case_study\\524942470472548352_claim_posts_pair.tsv", sep="\t")

    claims = list(example_df['s1'].values)
    claim_evi_pairs = []

    for i in range(len(example_df)):
        evi_list = example_df.iloc[i].to_list()[1:]
        evi_str = ' '.join(evi_list)
        claim_evi_pairs += [[claims[i], evi_str]]
    inps_e, msks_e, segs_e = tok2int_list_case_study(claim_evi_pairs, tokenizer, 130, max_seq_size=5)

    tokens_li = []
    for inps in inps_e:
        li = tokenizer.convert_ids_to_tokens(inps)
        tokens_li += [li]

    inputs = [[inps_e, msks_e, segs_e]]

    user_embed = torch.zeros(size=(args.evi_num, args.num_users, args.user_embed_dim))

    user_embeds = [user_embed]

    validset_reader = FNNDataLoader(None, tokenizer, args, inputs=inputs,
                                    # inputs_s=inputs_s_test,
                                    filenames=["politifact15631"],
                                    labels=[1],
                                    aux_info=None,
                                    batch_size=args.valid_batch_size, test=True)

    return validset_reader, tokens_li
