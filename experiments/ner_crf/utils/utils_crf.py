import crf
import torchcrf

from transformers import BertForTokenClassification


def stacking_crf_layer(model, use_impl=False):
    # assert isinstance(model, BertForTokenClassification)
    num_labels = model.classifier.out_features
    crf_layer = torchcrf.CRF(num_tags=num_labels, batch_first=True)
    if use_impl:
        crf_layer = crf.CRF(nb_labels=num_labels,
                            bos_tag_id=crf.Const.BOS_TAG_ID,
                            eos_tag_id=crf.Const.EOS_TAG_ID,
                            batch_first=True)
    model.add_module('CRFLayer', crf_layer)
    return model


if __name__ == "__main__":
    model = BertForTokenClassification.from_pretrained('monologg/kobert')
    model = stacking_crf_layer(model, True)
    print(model)
