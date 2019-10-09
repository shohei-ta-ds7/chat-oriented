# coding: utf-8


def get_dist_1(dial_list):
    response_list = [dial["inf"][0].split(" ") for dial in dial_list]
    vocab_list = [word for res in response_list for word in res]
    if len(vocab_list) < 1:
        return 0
    return len(set(vocab_list)) / len(vocab_list)


def get_dist_2(dial_list):
    response_list = [dial["inf"][0].split(" ") for dial in dial_list]
    SOS = "<sos>"
    EOS = "<eos>"
    response_list = [[SOS]+res+[EOS] for res in response_list]
    vocab_list = [(res[i], res[i+1]) for res in response_list for i in range(len(res)-1)]
    if len(vocab_list) < 1:
        return 0
    return len(set(vocab_list)) / len(vocab_list)
