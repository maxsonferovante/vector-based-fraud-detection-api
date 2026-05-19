# Vector-Based Fraud Detection API

Este projeto é uma API de detecção de fraude de alta performance escrita em C++, utilizando busca vetorial e otimizações SIMD.

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes dependências instaladas no seu macOS:

```bash
# Instalar compilador e ferramentas de build
xcode-select --install

# Instalar dependências via Homebrew
brew install cmake boost zlib
```

## Como Rodar no CLion (Recomendado)

O CLion utiliza o CMake para gerenciar as dependências e o processo de build automaticamente.

1. **Abrir o Projeto:**
   - Abra o CLion e selecione **Open**.
   - Navegue até a pasta do projeto e selecione-a.

2. **Configurar o CMake:**
   - O CLion deve carregar o `CMakeLists.txt` automaticamente.
   - Caso o Boost não seja encontrado, vá em `Settings` > `Build, Execution, Deployment` > `CMake`.
   - Em **CMake options**, adicione: `-DCMAKE_PREFIX_PATH=/opt/homebrew`
   - Clique em **Reload Changes**.

3. **Executar:**
   - No canto superior direito, selecione o target `vector_based_fraud_detection_api`.
   - Clique no botão verde **Run** (Play).

## Como Rodar via Terminal (CMake)

Se preferir compilar manualmente via terminal, utilize os comandos abaixo:

```bash
# Criar diretório de build
mkdir -p build && cd build

# Configurar o projeto
cmake ..

# Compilar
cmake --build .

# Executar
./vector_based_fraud_detection_api
```

## Otimizações de Performance (Opcional)

O projeto suporta **PGO (Profile-Guided Optimization)** e **LTO (Link-Time Optimization)** para performance máxima:

- **LTO:** Para ativar, use `-DCPP_LTO=ON` no CMake.
- **PGO:** 
  1. Compile com `-DPGO_GENERATE=ON`.
  2. Rode a aplicação para gerar dados de perfil.
  3. Recompile com `-DPGO_USE=ON`.

---

**Nota sobre o erro de 'boost/asio.hpp'**: Este erro ocorre ao tentar compilar o arquivo `main.cpp` isoladamente. Sempre utilize o sistema de build CMake (seja via CLion ou Terminal) para garantir que todas as bibliotecas e headers sejam vinculados corretamente.
