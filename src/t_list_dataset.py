import os
import os.path as osp
import json
from tqdm import tqdm

root_dir = 'H:/YoutubeGestureDataset/output-gfava-v0.3/paired_images'
video_list = [d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))]
train_video = video_list[:-100]
val_video = video_list[-100:-50]
test_video = video_list[-50:]

data_list = {
    'train': {},
    'val': {},
    'test': {},
}

for video_id in tqdm(video_list):
    video_dir = osp.join(root_dir, video_id)
    paired_list = [d for d in os.listdir(video_dir) if osp.isdir(osp.join(video_dir, d))]
    
    for paired_id in paired_list:
        data = {
            'gt_image': osp.join(video_id, paired_id, 'gt_image.png').replace('\\', '/'),
            'gt_mask': osp.join(video_id, paired_id, 'gt_mask.png').replace('\\', '/'),
            'ref_image': osp.join(video_id, paired_id, 'ref_image.png').replace('\\', '/'),
            'ref_mask': osp.join(video_id, paired_id, 'ref_mask.png').replace('\\', '/'),
            'refine_image': osp.join(video_id, paired_id, 'refine_image.png').replace('\\', '/'),
            'render_image': osp.join(video_id, paired_id, 'render_image.png').replace('\\', '/'),
        }

        is_valid = True
        for key in data.keys():
            fn = osp.join(root_dir, data[key])
            if not osp.exists(fn):
                print(f"File not found: {fn}")
                is_valid = False
            
        if is_valid:
            if video_id in train_video:
                data_list['train'][f"{video_id}.{paired_id}"] = data
            elif video_id in val_video:
                data_list['val'][f"{video_id}.{paired_id}"] = data
            elif video_id in test_video:
                data_list['test'][f"{video_id}.{paired_id}"] = data

with open(f'{root_dir}/paired_images.json', 'w') as fp:
    json.dump(data_list, fp)

pass

