import os
import sys
from typing import List
from core.hill.hill import *
from core.matrizes.matrizes import *

def limpar():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    limpar()
    print("Bem vindo ao programa de Criptografia com Matriz 2×2 (Cifra de HillSimples)\n")

def explicacao():
    while True:
        limpar()
        print ("A cifra de Hill é um método de criptografia clássica que utiliza álgebra linear " \
                "para transformar blocos de letras em blocos cifrados. Em vez de substituir letra por letra, "\
                "como ocorre em cifras simples, a cifra de Hill trabalha com vetores e matrizes.\n\n")
        print("O processo funciona assim:")
        print("1- Cada letra do alfabeto é convertida em um número (A=0, B=1, ... Z=25).")
        print("2- Agrupa-se em blocos do tamanho da matriz (ex: 2×2).")
        print("3- Converte cada bloco em vetor e multiplica pela matriz-chave (módulo 26).")
        print("4- Converte o resultado de volta para letras.\n")
        escolha = input("Se quiser voltar ao menu digite 0, para sair do programa digite 3: ").strip()
        if escolha == "0":
            return 
        if escolha == "3":
            print("Saindo...")
            sys.exit(0)
        print("Valor inválido. Tente novamente.")
        input("Pressione Enter para continuar...")

def main():
    while True:
        menu()
        questao = input("Digite um número:\n1 - Ir ao programa\n2 - Explicação\n3 - Fechar o programa\n\n> ").strip()

        if questao == "1":
            modulo = int(input("Digite o módulo da criptografia: ").strip())
            ordem = int(input("Digite a ordem da matriz: ").strip())
            text = input(f"Digite todos os valores da matriz separados por espaço: ")
            matriz: List[int | float] | List[List[int | float]]

            valores = [float(v) for v in text.split()]
            matriz = [valores[i*ordem:(i+1)*ordem] for i in range(ordem)]

            matriz2=converter_para_numpy(matriz)
            obter_inverso_modular(matriz2, modulo)
            
            texto=input("Digite o texto a ser criptografado: ")
            texto_crip=criptografar_hill(texto, matriz2)
            texto_descrip=decriptografar_hill(texto_crip, matriz2)



            print("\nMensagem original: ", texto)
            print("Mensagem criptograda: ", texto_crip)
            print("Mensagem descriptograda: ",texto_descrip)
            input("\nPressione Enter para continuar...")

        elif questao == "2":
            explicacao()

        elif questao == "3":
            print("Fechando o programa...")
            break

        else:
            print("Opção inválida. Digite 1, 2 ou 3.")
            input("Pressione Enter para tentar novamente...")

if __name__ == "__main__":
    main()
