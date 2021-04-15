#!/bin/sh

### UTILS FUNCTIONS ###
download_file () {
    if [ -e "$3/$1" ]; then
        echo "$3/$1 exists"
    else
        echo "Downloading $1..."
        curl "$2" --create-dirs -o "$3/$1"
    fi
}

## Download the datasets
# Download the FB15k-237 dataset
echo "Checking the FB15k-237 dataset..."
download_file "entities.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/entities.dict" "data/datasets/FB15k-237"
download_file "relations.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/relations.dict" "data/datasets/FB15k-237"
download_file "test.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/test.txt" "data/datasets/FB15k-237"
download_file "train.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt" "data/datasets/FB15k-237"
download_file "valid.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/valid.txt" "data/datasets/FB15k-237"
download_file "README.md" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/README.txt" "data/datasets/FB15k-237"

# Download the wn18rr dataset
echo "Checking the wn18rr dataset..."
download_file "entities.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/entities.dict" "data/datasets/wn18rr"
download_file "relations.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/relations.dict" "data/datasets/wn18rr"
download_file "test.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/test.txt" "data/datasets/wn18rr"
download_file "train.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/train.txt" "data/datasets/wn18rr"
download_file "valid.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/valid.txt" "data/datasets/wn18rr"
download_file "README.md" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/README.txt" "data/datasets/wn18rr"

# Download the YAGO3-10 dataset
echo "Checking the YAGO3-10 dataset..."
download_file "entities.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/entities.dict" "data/datasets/YAGO3-10"
download_file "relations.dict" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/relations.dict" "data/datasets/YAGO3-10"
download_file "test.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt" "data/datasets/YAGO3-10"
download_file "train.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt" "data/datasets/YAGO3-10"
download_file "valid.txt" "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt" "data/datasets/YAGO3-10"

# Create virtual environment
if [-e "venv" ]; then
    echo "The virtual environment already exists"
    source venv/bin/activate
else
    echo "Creating a new virtual environment"
    python3 -m venv venv
    source venv/bin/activate
fi

# Install the required libraries
pip3 install -r requirements.txt


