import os.path

from speechcraft.core.generation import preload_models, load_codec_model
from speechcraft.settings import MODELS_DIR, USE_GPU
from speechcraft.supp.utils import get_cpu_or_gpu

def get_hubert_manager_and_model(install_path: str = None):
    # Imports need to be here to avoid "circular" import error.
    # The reason in particular is not as clear. But works this way.
    from speechcraft.voice_cloning.custom_tokenizer import CustomTokenizer
    from speechcraft.voice_cloning.hubert_manager import HuBERTManager
    from speechcraft.voice_cloning.pre_kmeans_hubert import CustomHubert

    if install_path is None:
        install_path = MODELS_DIR

    # large huber pair
    huber_model_name = 'hubert_base_ls960_23.pth'
    hubert_model_path = os.path.join(install_path, huber_model_name)

    # small huber pair
    # huber_model_name = 'quantifier_V1_hubert_base_ls960_14.pth'

    # f"{tokenizer_lang}_tokenizer.pth"
    tokenizer_model_path = os.path.join(install_path, f"tokenizer_{huber_model_name}")

    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed(model_path=hubert_model_path)
    hubert_manager.make_sure_tokenizer_installed(model=huber_model_name,
                                                 local_tokenizer_path=tokenizer_model_path)

    device = get_cpu_or_gpu()

    hubert_model = CustomHubert(checkpoint_path=hubert_model_path).to(device)
    meta_encodec_model = load_codec_model(use_gpu=True)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint(tokenizer_model_path, map_location=device)

    return hubert_manager, hubert_model, meta_encodec_model, tokenizer


def make_sure_models_are_downloaded(install_path: str = None):
    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer

    # download and load all models
    preload_models(
        text_use_gpu=USE_GPU,
        text_use_small=False,
        coarse_use_gpu=USE_GPU,
        coarse_use_small=False,
        fine_use_gpu=USE_GPU,
        fine_use_small=False,
        codec_use_gpu=USE_GPU,
        force_reload=False,
        path=install_path
    )


def download_all_models_init(install_path: str = None):
    if install_path is None:
        install_path = MODELS_DIR

    # create models folder if not exists
    if not os.path.isdir(install_path):
        os.makedirs(install_path, exist_ok=True)

    # files for voice cloning
    if not os.path.isfile(os.path.join(install_path, 'hubert_base_ls960_23.pth')):
        get_hubert_manager_and_model(install_path=install_path)

    # download files for speechcraft
    if not os.path.isfile(os.path.join(install_path, 'coarse_2.pt')):
        make_sure_models_are_downloaded(install_path=install_path)


