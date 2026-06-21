"""
feature_extraction.py
======================
Extração de features para CBIR no dataset ISIC 2018 (lesões de pele).

Famílias de features implementadas:
    1. Cor      -> histogramas em HSV e CIE-Lab (+ normalização de iluminação)
    2. Textura  -> banco de filtros de Gabor (média e variância por filtro)
                   + GLCM/Haralick e LBP como complemento opcional
    3. Forma    -> calculada a partir da MÁSCARA de segmentação (ISIC Task 1):
                   área, perímetro, compacidade, excentricidade, solidez,
                   assimetria e irregularidade de borda (mapeia o A e B do ABCD)
    4. Deep     -> embedding da penúltima camada de uma ResNet50 (ImageNet)

Dependências:
    pip install numpy opencv-python scikit-image scipy pillow torch torchvision pandas

Observação: este módulo só EXTRAI e normaliza as features. A parte de
fusão, métrica de similaridade e avaliação (Precision@k, mAP) é a próxima etapa.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import gabor_kernel
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ---------------------------------------------------------------------------
# 1. COR
# ---------------------------------------------------------------------------

def shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """Normalização de iluminação (color constancy) pela norma de Minkowski.

    Reduz a influência do dispositivo/iluminação na cor, importante no ISIC
    porque as imagens vêm de equipamentos diferentes. p=1 equivale ao Gray World.
    """
    img = img_bgr.astype(np.float32)
    # norma de Minkowski por canal
    norm = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    norm[norm == 0] = 1.0
    gain = norm.mean() / norm
    out = np.clip(img * gain, 0, 255).astype(np.uint8)
    return out


def color_histogram(img_bgr: np.ndarray,
                    bins: tuple[int, int, int] = (8, 8, 8),
                    normalize_illumination: bool = True) -> np.ndarray:
    """Histograma de cor concatenado em HSV e Lab, normalizado (soma = 1).

    HSV separa cromaticidade de luminância; Lab é perceptualmente uniforme.
    Variação de cor é sinal clínico (o 'C' do ABCD), por isso usamos os dois.
    """
    if normalize_illumination:
        img_bgr = shades_of_gray(img_bgr)

    feats = []
    for conv in (cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2Lab):
        space = cv2.cvtColor(img_bgr, conv)
        hist = cv2.calcHist([space], [0, 1, 2], None, bins,
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # L1 -> robusto a escala
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. TEXTURA
# ---------------------------------------------------------------------------

def build_gabor_kernels(frequencies=(0.1, 0.2, 0.3, 0.4),
                        n_orientations: int = 6) -> list[np.ndarray]:
    """Banco de filtros de Gabor (parte real) variando frequência e orientação."""
    kernels = []
    for theta in np.linspace(0, np.pi, n_orientations, endpoint=False):
        for freq in frequencies:
            kernel = np.real(gabor_kernel(freq, theta=theta))
            kernels.append(kernel)
    return kernels


def gabor_features(gray: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Para cada filtro, retorna [média, variância] da resposta -> vetor estável."""
    feats = []
    g = gray.astype(np.float32) / 255.0
    for kernel in kernels:
        filtered = ndi.convolve(g, kernel, mode="wrap")
        feats.extend([filtered.mean(), filtered.var()])
    return np.asarray(feats, dtype=np.float32)


def glcm_features(gray: np.ndarray,
                  distances=(1, 3),
                  angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> np.ndarray:
    """Descritores de Haralick (GLCM): contraste, dissimilaridade, homogeneidade,
    energia, correlação e ASM. Clássico em dermatoscopia, complementa o Gabor."""
    g = (gray / 4).astype(np.uint8)  # quantiza para 64 níveis (acelera)
    glcm = graycomatrix(g, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)
    props = ("contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM")
    feats = [graycoprops(glcm, p).ravel() for p in props]
    return np.concatenate(feats).astype(np.float32)


def lbp_features(gray: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """Histograma de Local Binary Patterns (uniform) -> textura local barata."""
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins),
                           density=True)
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. FORMA  (precisa da máscara de segmentação do ISIC Task 1)
# ---------------------------------------------------------------------------

def shape_features(mask: np.ndarray) -> np.ndarray:
    """Descritores de forma da maior região da máscara binária.

    Retorna: [compacidade, excentricidade, solidez, extent,
              diâmetro_equiv_norm, assimetria, irregularidade_borda].
    Assimetria e irregularidade de borda correspondem ao A e B do ABCD.
    """
    n_feats = 7
    mask_bin = (mask > 0).astype(np.uint8)
    labeled = measure.label(mask_bin)
    props = measure.regionprops(labeled)
    if not props:
        return np.zeros(n_feats, dtype=np.float32)

    region = max(props, key=lambda r: r.area)
    area = float(region.area)
    perim = float(region.perimeter) if region.perimeter > 0 else 1.0

    compactness = (4.0 * np.pi * area) / (perim ** 2)        # 1.0 = círculo
    eccentricity = float(region.eccentricity)
    solidity = float(region.solidity)                        # área/convex hull
    extent = float(region.extent)                            # área/bbox
    eq_diam = float(region.equivalent_diameter) / np.sqrt(area)  # normalizado

    # --- Assimetria: roda a lesão para os eixos principais e dobra ---
    asymmetry = _asymmetry(region)

    # --- Irregularidade de borda: razão perímetro/perímetro do convex hull ---
    hull_perim = _convex_perimeter(region)
    border_irreg = perim / hull_perim if hull_perim > 0 else 1.0

    return np.array([compactness, eccentricity, solidity, extent,
                     eq_diam, asymmetry, border_irreg], dtype=np.float32)


def _asymmetry(region) -> float:
    """Fração de área não sobreposta ao espelhar a lesão nos eixos principais."""
    sub = region.image.astype(np.uint8)
    angle = np.degrees(region.orientation)
    rotated = ndi.rotate(sub, -angle, reshape=True, order=0)
    rotated = (rotated > 0).astype(np.uint8)

    scores = []
    for axis in (0, 1):
        flipped = np.flip(rotated, axis=axis)
        diff = np.logical_xor(rotated, flipped).sum()
        total = rotated.sum() * 2
        scores.append(diff / total if total > 0 else 0.0)
    return float(np.mean(scores))


def _convex_perimeter(region) -> float:
    hull = region.convex_image.astype(np.uint8)
    contours, _ = cv2.findContours(hull, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
    return float(cv2.arcLength(contours[0], True))


# ---------------------------------------------------------------------------
# 4. DEEP FEATURES (ResNet50 pré-treinada)
# ---------------------------------------------------------------------------

class DeepExtractor:
    """Embedding de 2048-d da penúltima camada da ResNet50 (ImageNet).

    Carregado de forma preguiçosa para o módulo funcionar mesmo sem torch
    instalado, caso você queira usar só as features clássicas.
    """

    def __init__(self, device: str | None = None):
        import torch
        from torchvision import models
        from torchvision.models import ResNet50_Weights

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()      # saída = 2048-d
        self.model = model.eval().to(self.device)
        self.preprocess = weights.transforms()

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        with self.torch.no_grad():
            x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            feat = self.model(x).cpu().numpy().ravel()
        return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# PIPELINE: extrai todas as features de uma imagem
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    use_color: bool = True
    use_gabor: bool = True
    use_glcm: bool = True
    use_lbp: bool = True
    use_shape: bool = True
    use_deep: bool = True
    resize: tuple[int, int] = (256, 256)
    gabor_kernels: list = field(default_factory=build_gabor_kernels)


def extract_all(image_path: str,
                mask_path: str | None,
                cfg: FeatureConfig,
                deep: DeepExtractor | None = None) -> dict[str, np.ndarray]:
    """Extrai cada bloco de features e devolve um dict {nome: vetor}.

    Manter separado por bloco é o que permite a FUSÃO TARDIA depois
    (normalizar e combinar cada modalidade com pesos próprios).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_bgr = cv2.resize(img_bgr, cfg.resize)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blocks: dict[str, np.ndarray] = {}

    if cfg.use_color:
        blocks["color"] = color_histogram(img_bgr)
    if cfg.use_gabor:
        blocks["gabor"] = gabor_features(gray, cfg.gabor_kernels)
    if cfg.use_glcm:
        blocks["glcm"] = glcm_features(gray)
    if cfg.use_lbp:
        blocks["lbp"] = lbp_features(gray)

    if cfg.use_shape:
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, cfg.resize, interpolation=cv2.INTER_NEAREST)
            blocks["shape"] = shape_features(mask)
        else:
            blocks["shape"] = np.zeros(7, dtype=np.float32)

    if cfg.use_deep:
        if deep is None:
            deep = DeepExtractor()
        pil = Image.open(image_path).convert("RGB")
        blocks["deep"] = deep.extract(pil)

    return blocks


# ---------------------------------------------------------------------------
# CONSTRÓI A BASE DE FEATURES (percorre o diretório do ISIC)
# ---------------------------------------------------------------------------

def find_mask(image_path: str, masks_dir: str | None) -> str | None:
    """Casa a imagem com sua máscara do ISIC Task 1.
    Padrão típico: ISIC_0000000.jpg  ->  ISIC_0000000_segmentation.png
    """
    if not masks_dir:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidate = os.path.join(masks_dir, f"{stem}_segmentation.png")
    return candidate if os.path.exists(candidate) else None


def build_feature_database(images_dir: str,
                           masks_dir: str | None = None,
                           cfg: FeatureConfig | None = None,
                           out_path: str = "isic_features.npz"):
    """Extrai features de todas as imagens e salva matrizes por bloco em .npz.

    Salvar por bloco (em vez de tudo concatenado) facilita normalizar e
    testar pesos diferentes na fusão sem reextrair nada.
    """
    cfg = cfg or FeatureConfig()
    deep = DeepExtractor() if cfg.use_deep else None

    paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Encontradas {len(paths)} imagens.")

    ids: list[str] = []
    collected: dict[str, list[np.ndarray]] = {}

    for i, p in enumerate(paths, 1):
        mask = find_mask(p, masks_dir)
        try:
            blocks = extract_all(p, mask, cfg, deep)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        ids.append(os.path.splitext(os.path.basename(p))[0])
        for name, vec in blocks.items():
            collected.setdefault(name, []).append(vec)
        if i % 50 == 0:
            print(f"  {i}/{len(paths)} processadas")

    arrays = {name: np.vstack(vecs) for name, vecs in collected.items()}
    arrays["ids"] = np.array(ids)
    np.savez_compressed(out_path, **arrays)
    print(f"Salvo em {out_path}. Blocos: "
          f"{[(k, v.shape) for k, v in arrays.items() if k != 'ids']}")
    return out_path


# ---------------------------------------------------------------------------
# NORMALIZAÇÃO PARA FUSÃO (z-score por bloco -> próximo passo: similaridade)
# ---------------------------------------------------------------------------

def zscore_normalize(matrix: np.ndarray, eps: float = 1e-8):
    """Padroniza colunas (média 0, desvio 1). Retorna (matriz, mean, std)
    para você aplicar a mesma transformação na imagem de consulta."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + eps
    return (matrix - mean) / std, mean, std


if __name__ == "__main__":
    # Exemplo de uso (ajuste os caminhos do ISIC 2018):
    #   build_feature_database(
    #       images_dir="ISIC2018_Task3_Training_Input",
    #       masks_dir="ISIC2018_Task1_Training_GroundTruth",  # opcional
    #       out_path="isic_features.npz",
    #   )
    print("Edite os caminhos em __main__ e rode: python feature_extraction.py")
