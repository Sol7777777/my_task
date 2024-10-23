# task3

## 前备知识

### Self-Attention

Self-Attention（自注意力机制）是一种用于计算序列中各个位置之间关系的机制。它通过计算输入序列中每个元素与其他元素之间的相似性，生成加权表示。自注意力机制的核心在于能够动态地关注输入序列中重要的信息，而不是依赖固定的局部感受野。

在自注意力机制中，每个输入元素都会生成三个向量：查询（Query）、键（Key）和值（Value）。通过计算查询与所有键的相似度，生成一个权重分布，然后用这个权重对值进行加权求和，得到最终的输出。这种机制使得模型能够捕捉到长距离的依赖关系，适用于序列数据的处理。

### Multi-Head Attention

Multi-Head Attention（多头注意力）是自注意力机制的扩展。它通过并行计算多个注意力头，能够捕捉输入序列中不同部分的关系。每个注意力头独立地学习输入的不同特征，最后将这些头的输出拼接起来，形成更丰富的表示。

多头注意力的优势在于能够同时关注输入的不同子空间，从而增强模型的表达能力。通过这种方式，模型可以学习到更复杂的特征，有助于提升性能。

### Transformer

Transformer是由Vaswani等人在2017年提出的一种新型神经网络架构，最初应用于自然语言处理（NLP）任务。Transformer的核心思想是完全基于自注意力机制，而不依赖于递归或卷积结构。这使得Transformer能够有效处理长序列数据，并具有良好的并行计算能力。

Transformer架构由编码器和解码器组成，其中编码器负责输入序列的表示学习，解码器用于生成输出序列。在编码器中，每一层都包含多头自注意力机制和前馈神经网络。通过堆叠多个编码器层，Transformer能够捕捉到输入数据的复杂特征。

## ViT的结构组成

ViT（Vision Transformer）将Transformer架构直接应用于图像分类任务。与传统的卷积神经网络不同，ViT采用了一种新的输入表示方式，将图像分割成固定大小的补丁。ViT的结构主要由以下三个部分组成：Patch Embedding、Class Embedding和Position Embedding。

### 1. Patch Embedding

在ViT中，输入图像首先被分割成多个固定大小的补丁（例如16x16像素）。每个补丁被展平并线性嵌入，形成一个一维向量。这个过程将图像的空间信息转换为序列数据，使得ViT能够利用自注意力机制处理图像。

具体来说，假设输入图像的大小为 \( H \times W \times C \)（高度、宽度和通道数），则将其分割成 \( N \) 个补丁，每个补丁的大小为 \( P \times P \)。经过展平和线性映射后，补丁的嵌入向量的维度为 \( D \)（Transformer的维度）。这些嵌入向量将作为Transformer的输入序列。

### 2. Class Embedding

为了进行图像分类，ViT引入了一个额外的“分类嵌入”（Class Embedding）。这个嵌入是一个可学习的向量，类似于NLP中的[CLS]标记。分类嵌入被添加到补丁嵌入序列的开头，目的是在经过Transformer编码器后，能够通过这个嵌入来获得整个图像的表示。

在模型的训练过程中，分类嵌入会随着其他参数一起更新，从而学习到图像的全局特征。最终，经过Transformer编码器处理后，分类嵌入的输出将用于进行图像分类。

### 3. Position Embedding

由于Transformer本身并不具备处理序列顺序的能力，因此ViT需要引入位置编码（Position Embedding）。位置编码是一个与补丁嵌入向量相同维度的向量，用于表示每个补丁在图像中的位置。通过将位置编码与补丁嵌入相加，ViT能够保留空间信息。

ViT通常使用可学习的1D位置编码，表示每个补丁在输入序列中的位置。位置编码的引入使得模型能够理解补丁之间的相对位置关系，从而更好地捕捉图像的结构信息。

## 创新点

ViT的提出带来了多个创新点，使其在计算机视觉领域引起了广泛关注：

1. **直接应用变换器于图像**：ViT将标准的变换器架构直接应用于图像分类任务，挑战了传统卷积神经网络的主导地位。这一创新使得研究者们重新思考视觉任务的建模方式。

2. **图像补丁作为输入**：通过将输入图像分割成固定大小的补丁，ViT将图像视为序列数据。这种方法使得模型能够利用自注意力机制捕捉图像中的长距离依赖关系。

3. **减少卷积操作的依赖**：ViT通过自注意力机制捕捉图像特征，减少了对卷积操作的依赖。这一变化使得模型在处理图像时具有更大的灵活性。

4. **可扩展性**：ViT在大型数据集上进行预训练，展示了在多个视觉任务上的有效性和良好的迁移学习性能。通过在大规模数据集上进行训练，ViT能够学习到更丰富的特征表示。

5. **性能与计算效率**：在多个图像分类任务上，ViT表现出色，尤其是在大型数据集上预训练后，以较低的计算资源实现高性能。这一特性使得ViT在实际应用中具有良好的可行性。

## 实验结果

ViT在多个标准数据集上的实验结果表明，其性能优于传统的卷积神经网络。以下是一些关键的实验结果：

1. **ImageNet**：在ImageNet数据集上，ViT模型在多个变体上均取得了优异的分类性能。例如，ViT-L/16模型在ImageNet上达到了88.55%的准确率，优于许多传统的卷积神经网络。

2. **VTAB（Visual Task Adaptation Benchmark）**：在VTAB基准测试中，ViT在不同任务上的迁移学习性能表现出色，尤其是在Natural和Structured任务上，超越了许多现有的最先进方法。

3. **计算效率**：ViT在多个任务中展示了良好的性能与计算效率的平衡。在相同的计算预算下，ViT能够实现与传统卷积网络相当甚至更好的性能。

## 未来研究方向

尽管ViT在计算机视觉领域取得了显著的成功，但仍有许多研究方向值得探索：

1. **自监督学习**：自监督学习是一种利用未标注数据进行预训练的方法。未来的研究可以探讨如何将自监督学习与ViT结合，以进一步提升模型的性能。

2. **模型压缩与加速**：ViT模型通常较大，计算开销较高。研究者可以探索模型压缩和加速技术，以在保持性能的同时降低计算成本。

3. **多模态学习**：将ViT与其他模态（如文本、音频等）结合，进行多模态学习，将是一个有趣的研究方向。这可以帮助模型更好地理解复杂的场景和任务。

4. **领域适应**：研究如何使ViT在不同领域（如医疗影像、卫星图像等）中表现良好，提升其迁移学习能力。

5. **可解释性**：探索ViT模型的可解释性，以帮助研究者理解模型的决策过程，提升其在实际应用中的可信度。

## 实验

### 模型下载

刚开始使用timm库去下载vit的模型，但是貌似某次升级后国内无法直接连接Hugging Face网站，下载会报错。于是在网上搜索timm手动下载模型的方法，获取url，手动下载

```py
model = timm.create_model('convnext_base', num_classes=2, global_pool='')

print(model.default_cfg)
```

但是我获取的url是空的。。

```bash
{'url': '', 'hf_hub_id': 'timm/beit_base_patch16_384.in22k_ft_in22k_in1k', 'architecture': 'beit_base_patch16_384', 'tag': 'in22k_ft_in22k_in1k', 'custom_load': False, 'input_size': (3, 384, 384), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 1.0, 'crop_mode': 'center', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'num_classes': 1000, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head'}
```

于是只能换其他方式下载模型

尝试从torchvision.models中下载

torchvision.models为我们提供了几个不同的版本

1. **ViT-B/16**：基础版本，使用16x16的patch大小。
2. **ViT-B/32**：基础版本，使用32x32的patch大小。
3. **ViT-L/16**：较大版本，使用16x16的patch大小，具有更多的参数。
4. **ViT-L/32**：较大版本，使用32x32的patch大小。
5. **ViT-H/14**：更高性能的版本，具有更大的模型和更多的层次。

```py
from torchvision.models import vit_b_16
```

```py
model = vit_b_16(pretrained=False, num_classes=10)
```

因为cifar10是10分类问题，num_classes设置为10

如果把pretrain设置为True，说明使用预训练权重，那么则需要手动替换输出层

打印出模型得知：

```py
(heads): Sequential(
    (head): Linear(in_features=768, out_features=1000, bias=True)
  )
```

最后一层的输入是768，所以：

```py
model = vit_b_16(pretrained=True)
model.heads = nn.Linear(768, 10)
```

### 训练

大致上和之前的代码保持一致只需要做略微的修改

```py
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 改为ViT 输入大小
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),#同理
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
])
'''
'''

def train():
    model.train() 
    for epoch in range(num_epochs):
       '''
       '''
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def test():
    model.eval() 
    '''
    '''
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()

```

但是云端环境给的NVIDIA A10的GPU好像不太行，半天只算出了一轮的结果

```py
Files already downloaded and verified
Files already downloaded and verified
Epoch [1/50], Loss: 1.8614, Accuracy: 29.76%
```

理论上来说，只要下载好对应的模型，将

```py
model = vit_b_16(pretrained=True)
```

例如改为

```py
model = vit_b_18(pretrained=True)
```

就可以体验不同patch大小的vit模型带来的不同效果

止步于此。