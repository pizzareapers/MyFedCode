from network import vit_with_adapter

def GetNetwork(args, num_classes, shared_adapter_down, shared_adapter_up, **kwargs):

    if args.model == 'vit_b16':
        dual_model = vit_with_adapter.DualAdapterViT(shared_adapter_down=shared_adapter_down, shared_adapter_up=shared_adapter_up,
                                   num_classes=num_classes)
        single_model = vit_with_adapter.SingleAdapterViT(shared_adapter_down=shared_adapter_down,
                                                     shared_adapter_up=shared_adapter_up, num_classes=num_classes)
        feature_level = 2048

    else:
        raise ValueError("The model is not support")

    return dual_model, single_model, feature_level
