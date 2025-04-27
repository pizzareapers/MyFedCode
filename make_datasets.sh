# Set root directory (Please put the root directory here)
ROOT_DIR=

# Set the dataset locations
cat > $ROOT_DIR/FedSDAF/configs/default.py <<EOF
pacs_path = '$ROOT_DIR/FedSDAF/datasets/pacs/'
officehome_path = '$ROOT_DIR/FedSDAF/datasets/office_home/'
vlcs_path = '$ROOT_DIR/FedSDAF/datasets/vlcs/'
domainnet_path = '$ROOT_DIR/FedSDAF/datasets/domainnet/'
log_count_path = '$ROOT_DIR/FedSDAF/log/'
EOF

# Make label split for officehome and vlcs
python -u $ROOT_DIR/FedSDAF/data_loader/split_label.py \
    --root_path "$ROOT_DIR/FedSDAF/datasets/office_home" \
    --dataset 'officehome'
python -u $ROOT_DIR/FedSDAF/data_loader/split_label.py \
    --root_path "$ROOT_DIR/FedSDAF/datasets/vlcs" \
    --dataset 'vlcs'