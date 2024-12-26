from .imports import *
from concurrent.futures import ThreadPoolExecutor

# https://github.com/davidtvs/pytorch-lr-finder/blob/master/examples/mnist_with_amp.py
def simple_timer(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        func(*args, **kwargs)
        print('--- Time taken from {}: {} seconds'.format(
            func.__qualname__, time.time() - st
        ))
    return wrapper

# https://github.com/davidtvs/pytorch-lr-finder/blob/master/examples/mnist_with_amp.py
def conceal_stdout(enabled):
    if enabled:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f
    else:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

# https://forums.fast.ai/t/accumulating-gradients/33219/28
def reset_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def find_files(folder,ext='.mp4'):
    hits = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(ext):
                hits.append(os.path.join(root, file))
    return hits

def read_video(path,vfps=24,afps=16_000):
    video = VideoFileClip(path).with_fps(vfps)
    frames = [frame for frame in video.iter_frames()]
    audio = []
    for chunk in video.audio.with_fps(afps).iter_chunks(5000):
        audio.extend(chunk)
    audio = np.array(audio)
    audio = audio.mean(-1)
    return frames, audio

# read json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# write json
def write_json(path,data):
    with open(path,'w') as f:
        json.dump(data,f)


def zip_dir(folder_path, zip_path):
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zf.write(os.path.join(root, file), 
                         arcname=os.path.relpath(os.path.join(root, file), folder_path))

def unzip_file(zip_path, extract_path):
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_path)

def untar_file(tar_path, extract_path="."):
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_path)

def parallel_map(func, data, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, data))

def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


__all__ = [ "simple_timer",
            "conceal_stdout",
            "reset_seed",
            "find_files", 
            "read_video",
            "read_json", 
            "write_json", 
            "zip_dir", 
            "unzip_file", 
            "untar_file",
            "parallel_map",
            "clear_gpu_cache"]