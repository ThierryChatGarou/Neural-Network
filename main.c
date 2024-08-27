//For doubles use %lf in printf and scanf
//For float use just %f
//If supported, long float would be %Lf

#include <stdio.h>
#include <stdlib.h>     /* srand, rand, malloc*/
#include <time.h>       /* time */
#include <math.h>
void getch() {}

float alpha = 0.5; //0.5;
float VelocidadAprendizaje   = 0.00001;  //Alrededor de 0.1

struct Neurona
    {
    float salida;
    int numEntradas;
    float *entradas;  //cada entrada tiene un peso //usar malloc? //float peso[numEntradas]
    float *deltaentradas;  //deltapeso
    float gradiente;  //para el aprendizaje
    };

struct Capa
    {
    int numNeuronas;
    struct Neurona **listaNeuronas;  //Nota: C s¢lo permite un solo arreglo de este tipo al final de la estructura
    };

struct Red
    {
    int numCapas;
    struct Capa **listaCapas;
    float errorpromedioreciente;
    float factordesuavizadoreciente;
    float error;
    };


struct Neurona *neurona_crear(int numEntradas)
    {
    int i;
    struct Neurona *mNeurona = (struct Neurona *) malloc(sizeof(struct Neurona));
    mNeurona->numEntradas = numEntradas;
    mNeurona->entradas = (float *) malloc(sizeof(float) * numEntradas);
    mNeurona->deltaentradas = (float *) malloc(sizeof(float) * numEntradas);
    for(i=0; i<numEntradas; i++)
        {
        mNeurona->entradas[i]=(float)rand()/RAND_MAX;  //0 a 1
        mNeurona->deltaentradas[i]=0.0;
        }
    return mNeurona;
    }


void neurona_print(struct Neurona *mNeurona)
    {
    int i;
    printf("N: entradas=%d: ",mNeurona->numEntradas);
    for(i=0; i<mNeurona->numEntradas; i++)
        {
        printf("%1.2f,",mNeurona->entradas[i]);
        }
    printf(" salida=%1.2f\r\n",mNeurona->salida);
    }


float neurona_ftransfer(float entrada)
    {
    float salida;
    //salida=entrada;//DEBUG
    salida=tanh(entrada);
    //salida=sigmoid(entrada);
    return salida;
    }


float neurona_ftransferderivada(float entrada)
    {
    float salida;
    salida= 1.0 - entrada * entrada;  //aproximaci¢n para derivada de tanh
    //la derivada correcta es:
    //1-tanh^2 x
    //float th = tanh(entrada);
    //salida = 1.0 - th*th;
    return salida;
    }


void neurona_calculargradientedesalida(struct Neurona *mNeurona, float valorObjetivo)
    {
    float delta = valorObjetivo - mNeurona->salida;
    mNeurona->gradiente = delta * neurona_ftransferderivada(mNeurona->salida);  ////posible correccion desde el foro: valor de entradade neurona
    }


float neurona_suma_de_errores(int numero_de_Neurona, struct Capa *capaSiguiente)
    {
    int i;
    float sum =0.0;  //suma de los errores que  contribuimos a todas las neuronas que alimentamos
    for(i=0; i<capaSiguiente->numNeuronas-1; i++) //por cada neurona en la siguiente capa excluyendo el bias
        {
        struct Neurona *mNeu = capaSiguiente->listaNeuronas[i];
        sum += mNeu->entradas[numero_de_Neurona] * mNeu->gradiente;
        }
    return sum;
    }


void neurona_calculargradienteoculto(struct Neurona *mNeurona, int numero_de_Neurona, struct Capa *capaSiguiente)
    {
    float sumaDeErrores= neurona_suma_de_errores(numero_de_Neurona, capaSiguiente);
    mNeurona->gradiente = sumaDeErrores * neurona_ftransferderivada(mNeurona->salida);   //correccion desde el foro: valor de entradade neurona
    }


void neurona_actualizar_pesos(struct  Neurona *mNeurona, struct Capa *capaAnterior)  //capaAnterior: capa anterior a la neurona que se calcula
    {
    int i;
    for(i=0; i<mNeurona->numEntradas; i++) // por cada peso en la neurona
        {
        float oldDeltaweight = mNeurona->deltaentradas[i];
        float salidaNeuronaCapaAnterior = capaAnterior->listaNeuronas[i]->salida;

//Entradas individuales, magnificadas por el gradiente y la velocidad de aprendizaje
//alpha: Agrega un momentum = una fracion del deltaweight anterior
        float newDeltaweight = VelocidadAprendizaje*salidaNeuronaCapaAnterior*mNeurona->gradiente + alpha*oldDeltaweight;
        mNeurona->deltaentradas[i] = newDeltaweight;
        mNeurona->entradas[i] += newDeltaweight;
        }
    }


struct Capa *capa_crear(int numEntradas, int numNeuronas)
    {
    int i;
    struct Capa *mCapa = (struct Capa *) malloc(sizeof(struct Capa));
    mCapa->numNeuronas=numNeuronas;
    mCapa->listaNeuronas = (struct Neurona **) malloc(sizeof(struct Neurona *) * numNeuronas);
    for(i=0; i<numNeuronas; i++)
        {
        mCapa->listaNeuronas[i] = neurona_crear(numEntradas);
        }
    mCapa->listaNeuronas[numNeuronas-1]->salida = 1.0;  //Bias
    for(i=0; i<numNeuronas; i++)
        {
        //neurona_print(mCapa->listaNeuronas[i]);
        }
    return mCapa;
    }


void capa_procesar_neurona(struct Capa *capaAnterior, struct Neurona *mNeurona)
    {
    int i;
    float sum=0.0;
    int numNeu=capaAnterior->numNeuronas;
    for(i=0; i<numNeu; i++)
        {
        struct Neurona *mN = capaAnterior->listaNeuronas[i];
        sum += mN->salida * mNeurona->entradas[i];
        }
    mNeurona->salida = neurona_ftransfer(sum);
    }

void capa_procesar_todo_alt(struct Capa *capaAnterior, struct Capa *capaActual)
    {
    int i;
    int numNeu = capaActual->numNeuronas;
    for(i=0; i<numNeu-1; i++)  //excepto Bias
        {
        struct Neurona *mN = capaActual->listaNeuronas[i];
        capa_procesar_neurona(capaAnterior, mN);
        }
    }


//Equivalente: Neuron::feedForward
void capa_procesar_todo(struct Capa *capaAnterior, struct Capa *capaActual)  //Forma alternativa, pero puede ser m s dificil de entender.
    {
    int i;
    int numNeuAct = capaActual->numNeuronas;
    int numNeuAnt = capaAnterior->numNeuronas;
    for(i=0; i<numNeuAct-1; i++)  //No modificamos la salida del bias
        {
        int j;
        float sum=0.0;
        for(j=0; j<numNeuAnt; j++)  //Leer todas las capas anteriores incluyendo el bias
            {
            sum += capaAnterior->listaNeuronas[j]->salida * capaActual->listaNeuronas[i]->entradas[j];
            }
        capaActual->listaNeuronas[i]->salida = neurona_ftransfer(sum);
        }
    }


void capa_print(struct Capa *mCapa)
    {
    int i;
    printf("NumNeu=%d: \r\n",mCapa->numNeuronas);

    for(i=0; i<mCapa->numNeuronas; i++)
        {
        neurona_print(mCapa->listaNeuronas[i]);
        }
    printf("\r\n");
    }


struct Red *red_crear(int *topologiaCapas, int numCapas)
    {
    int i;
    struct Red *mRed = (struct Red*) malloc(sizeof(struct Red));
    mRed->numCapas=numCapas;
    mRed->listaCapas = (struct Capa **) malloc(numCapas * sizeof(struct Capa *));
    //*mRed->listaCapas = (struct Capa*) malloc(sizeof(struct Capa) * numCapas);  //Las estructuras en C estan limitadas a un solo arreglo del tipo struct Capa *listadecapas[], entonces los cambi‚ por un doble puntero struct Capa **listadecapas
    mRed->listaCapas[0]=capa_crear(1,topologiaCapas[0]+1);
    //int *caracteristicas = (int *)calloc(topologiaCapas[0],sizeof(int));
    //mRed->caracteristicas=caracteristicas;
    for(i=1; i<numCapas; i++)
        {
        //printf("Capa %d\r\n",i);
        mRed->listaCapas[i]=capa_crear(topologiaCapas[i-1]+1,topologiaCapas[i]+1);  //+1 neurona polirarizadora
        }
    //Otros elementos opcionales:
    mRed->errorpromedioreciente=0.0;
    mRed->factordesuavizadoreciente=100000.0;
    mRed->error=0.0;
    return mRed;
    }


void red_print(struct Red *mRed)
    {
    int i;
    int numCapas=mRed->numCapas;
    for(i=0; i<numCapas; i++)
        {
        struct Capa *mCapa = mRed->listaCapas[i];
        capa_print(mCapa);
        }
    }



void red_alimentar(struct Red *mRed, float *valoresdelasentradas, int numerodeentradas)
    {
    int i;

    struct Capa *mCapa = mRed->listaCapas[0];
    int numNeu = mCapa->numNeuronas;
    if(numerodeentradas != numNeu-1)  //excepto bias
        {
        printf("Error aqui: numerodeentradas(%d) != numNeu-1(%d)",numerodeentradas,numNeu-1);
        getchar();
        }

    //alimentando la capa de entrada
    for(i=0; i<numerodeentradas; i++)
        {
        struct Neurona *nM = mCapa->listaNeuronas[i];
        nM->entradas[0]=valoresdelasentradas[i];  //No es necesario, sólo es para tener los valores de entradas como referencia.
        //nM->salida=caracteristicas_asignar(caracteristicas[i], valoresdelasentradas[i]);  //nM->salida=valoresdelasentradas[i];
        //nM->salida=red_calcular_caracteristicas(mRed->caracteristicas[i],valoresdelasentradas[i]);
        nM->salida=nM->entradas[0];
        }

    //procesar el resto de las capas
    int numCapas=mRed->numCapas;
    for(i=1; i<numCapas; i++)
        {
        capa_procesar_todo(mRed->listaCapas[i-1],mRed->listaCapas[i]);
        }
    }


void red_aprender(struct Red *mRed, float *valoresObjetivo, int numValoresObjetivo)  //backProp(targetVals)
    {
//calcular el error total de la red (RMS)
//obtener ultima capa
    int ultima = mRed->numCapas-1;
    struct Capa *ultimaCapa = mRed->listaCapas[ultima];
    int i;
    if(numValoresObjetivo != ultimaCapa->numNeuronas-1)
        {
        printf("Debug aqui aprender\r\n");
        }
    mRed->error=0.0;
    for(i=0; i<ultimaCapa->numNeuronas-1; i++) //por cada neurona en la ultima capa sin incluir el bias
        {
        float delta = valoresObjetivo[i] - ultimaCapa->listaNeuronas[i]->salida;
        mRed->error += delta*delta;
        }

    mRed->error = mRed->error / (ultimaCapa->numNeuronas-1);
    mRed->error = sqrt(mRed->error);  //RMS

//codigo para debugear el progreso de aprendizaje
    mRed->errorpromedioreciente =
        (mRed->errorpromedioreciente*mRed->factordesuavizadoreciente+mRed->error)/
        (mRed->factordesuavizadoreciente+1.0);

//calcular la gradiente de la capa de salida
    for(i=0; i<ultimaCapa->numNeuronas-1; i++)
        {
        neurona_calculargradientedesalida(ultimaCapa->listaNeuronas[i], valoresObjetivo[i]);
        }

//calcular gradiente de las capas oculta, empezando por la ultima capa
    int l;
    for(l=mRed->numCapas-2; l>0; l--)
        {
        struct Capa *capaOculta    = mRed->listaCapas[l];
        struct Capa *capaSiguiente = mRed->listaCapas[l+1];
        for(i=0; i<capaOculta->numNeuronas; i++ )   //DEBUG: se incluye el bias?
            {
            neurona_calculargradienteoculto(capaOculta->listaNeuronas[i], i, capaSiguiente);
            }
        }


//Actualizar los pesos de las connecciones
//desde la capa de salida hasta la primera capa oculta
    for(l=mRed->numCapas-1 ; l>0 ; l--)
        {
        struct Capa *capaOculta    = mRed->listaCapas[l];
        struct Capa *capaAnterior = mRed->listaCapas[l-1];
        for(i=0; i<capaOculta->numNeuronas-1; i++)  //por cada neurona en la capa oculta menos el bias
            {
            neurona_actualizar_pesos(capaOculta->listaNeuronas[i], capaAnterior);  //capaAnterior: capa anterior a la neurona que se calcula
            }
        }
    }


void red_resultados(struct Red *mRed)
    {
    int i;
    struct Capa *capaEnt = mRed->listaCapas[0];
    int numNeuEnt = capaEnt->numNeuronas;
    printf("Inputs(%d): ",numNeuEnt);
    for(i=0; i<numNeuEnt-1; i++)
        {
        struct Neurona *N = capaEnt->listaNeuronas[i];
        printf("%f,",N->salida);
        }
    int numCapas = mRed->numCapas;
    struct Capa *capaSal = mRed->listaCapas[numCapas-1];
    int numNeuSal = capaSal->numNeuronas;
    printf("\r\nOutputs(%d): ",numNeuSal);
    for(i=0; i<numNeuSal-1; i++)
        {
        printf("%f,",capaSal->listaNeuronas[i]->salida);
        }
    printf("\r\n\r\n");
    }


int archivo_fprintf(const char *format, ...);
int archivo_fscanf(const char *format, ...);
void archivo_nuevo(char *filename);
void archivo_abrir(char *filename);
void archivo_cerrar();

void red_guardar(struct Red *mRed, char *archivo)
    {
    archivo_nuevo(archivo);
    printf("Guardar en archivo %s\r\n",archivo);
    archivo_fprintf("Capas:%d\r\n",mRed->numCapas);
    archivo_fprintf("Topologia: ");
    int nC,nN,nE;
    for(nC=0; nC<mRed->numCapas; nC++)
        {
        struct Capa *mCapa = mRed->listaCapas[nC];
        archivo_fprintf("%d,",mCapa->numNeuronas-1);
        }
    archivo_fprintf("\r\n");

    //Guardar los valores de entradas
    for(nC=0; nC<mRed->numCapas; nC++)
        {
        struct Capa *mCapa = mRed->listaCapas[nC];
        archivo_fprintf("Capa %d:\r\n",nC);
        for(nN=0; nN<mCapa->numNeuronas-1; nN++)
            {
            struct Neurona *mNeurona = mCapa->listaNeuronas[nN];
            archivo_fprintf("\t%d-Entradas:%d\r\n\t\t",nN,mNeurona->numEntradas);
            for(nE=0; nE<mNeurona->numEntradas; nE++)
                {
                archivo_fprintf("%f,",mNeurona->entradas[nE]);
                }
            archivo_fprintf("\r\n");
            }
        }
    archivo_fprintf("\r\n\r\n");
    archivo_cerrar();
    }


struct Red *red_cargar(char *archivo)
    {
    struct Red *mRed;
    archivo_abrir(archivo);
    int numCapas;
    archivo_fscanf("Capas:%d\r\n",&numCapas);
    int nC,nN,nE;

    //cargar la topologia de la red y crear una nueva red.
    archivo_fscanf("Topologia: ");
    int topo[numCapas];
    for(nC=0; nC<numCapas; nC++)
        {
        int numNeu;
        archivo_fscanf("%d,",&numNeu);
        topo[nC]=numNeu;
        }
    mRed = red_crear(topo, numCapas);
    archivo_fscanf("\r\n");

    //cargar el valor de las entradas en cada neurona
    for(nC=0; nC<mRed->numCapas; nC++)
        {
        int nCapa;
        archivo_fscanf("Capa %d:\r\n",&nCapa);
        struct Capa *mCapa = mRed->listaCapas[nC];
        for(nN=0; nN<mCapa->numNeuronas-1; nN++)
            {
            struct Neurona *mNeurona = mCapa->listaNeuronas[nN];
            int numNeu,numEnt;
            archivo_fscanf("\t%d-Entradas:%d\r\n\t\t",&numNeu,&numEnt);
            if(nN!=numNeu || numEnt!=mNeurona->numEntradas)
                {
                printf("Debug needed here too!\r\n");
                }
            for(nE=0; nE<mNeurona->numEntradas; nE++)
                {
                archivo_fscanf("%f,",&mNeurona->entradas[nE]);
                }
            archivo_fscanf("\r\n");
            }
        }
    archivo_cerrar();
    return mRed;
    }



/*
void neural_network_in_c(int *a, int *b, int *c, int *d, int size, int layer, float bloque, float paisaje)
{
int i,j,k,l,m;
for(i=0;i<size;i++)
for(j=0;j<layer-2;j++)
{
if(a[i]>=b[j])
{
if(l[c[i]]>=m[d[j]])
{
int o=func();
}
else
{
if(l[c[i]]>=m[d[j]])
{


  if((o%16)!=0)  // comprobar que bloques necesitan actualizarse
    {
    bloque(o-(o%16),p-(p%16),paisaje[(p-(p%16))/16][(o-(o%16))/16]);  //actualizar bloque que esta arriba a la izquierda
    bloque(o-(o%16)+16,p-(p%16),paisaje[(p-(p%16))/16][(o+16-(o%16))/16]);  //actualizar bloque que esta arriba a la derecha
    if((p%16)!=0)
      {
      bloque(o-(o%16),p-(p%16)+16,paisaje[(p+16-(p%16))/16][(o-(o%16))/16]);  //actualizar bloque que esta abajo a la izquierda
      bloque(o-(o%16)+16,p-(p%16)+16,paisaje[(p+16-(p%16))/16][(o+16-(o%16))/16]);  //actualizar bloque que esta abajo a la derecha
      }
    }
  else if((p%16)!=0)
    {
    bloque(o-(o%16),p-(p%16),paisaje[(p-(p%16))/16][(o-(o%16))/16]);  //actualizar bloque que esta arriba a la izquierda
    bloque(o-(o%16),p-(p%16)+16,paisaje[(p+16-(p%16))/16][(o-(o%16))/16]);  //actualizar bloque que esta abajo a la izquierda
    }
  else
    {
    bloque(o,p,paisaje[p/16][o/16]);  //actualizar bloque que esta arriba a la izquierda, el bloque que esta en su posicion
    }
}
}
}*/

///================================================Administrador de operaciones (experimental)============================================


enum CARACTERISTICAS {IGUAL,ANTERIOR,DELTA_ANTERIOR,DELTA,ULTIMOS_10,INCREMENTO,CUADRADO,CUBO,SENO,COSENO,TANGENTE,SENO_H,COSENO_H,TANGENTE_H,LOG,LOG10,NOISE,TIME,RGB_SUM
                     };

//solo aplica a las entradas adicionales
struct Operacion
    {
    int opAd;         //Numero de operaciones adicionales agregadas
    int *operador;    //Operado que se aplica a la entrada. Igualdad, anterior, Suma, Seno, Coseno, etc.
    int *operandoA;   //Indica que entrada se usará como operando.
    int *operandoB;
    int *operandoC;
    int *operandoD;
    int maxOperadores;  //Numero de operadores reservados con malloc
    float **memorias;   //Almacena los valores previos de las entradas para algunos operadores
    char *primeraVez;    //indica que operador se ha ejecutado por primera vez.
    };

struct Operacion *operacion_crear_operadores(int maxOperadores)
    {
    struct Operacion *mOp = (struct Operacion *) malloc(sizeof(struct Operacion));
    mOp->opAd=0;
    mOp->maxOperadores=maxOperadores;
    mOp->operador = (int *) calloc(maxOperadores,sizeof(int));
    mOp->operandoA = (int *) calloc(maxOperadores,sizeof(int));
    mOp->operandoB = (int *) calloc(maxOperadores,sizeof(int));
    mOp->operandoC = (int *) calloc(maxOperadores,sizeof(int));
    mOp->operandoD = (int *) calloc(maxOperadores,sizeof(int));
    mOp->memorias = (float **) calloc(maxOperadores,sizeof(float *));
    //mOp->primeraVez = (char *) malloc(maxOperadores * sizeof(char));
    //memset(mOp->primeraVez,1,maxOperadores);
    return mOp;
    }

void _expandir_memoria_int(int **inPtr, int tamano)
    {
    int *ptr;
    ptr = (int *) realloc(*inPtr, sizeof(int) * tamano);
    if(ptr==NULL)
        {
        printf("Error Realloc line %d\r\n",__LINE__);
        getch();
        }
    else
        {
        *inPtr = ptr;
        }
    }

void _expandir_memoria_floatptr(float **inPtr, int tamano)
    {
    float **ptr;
    printf("_expandir_memoria_floatptr\r\n");
    ptr = (float **) realloc(*inPtr, sizeof(float *) * tamano);
    if(ptr==NULL)
        {
        printf("Error Realloc line %d\r\n",__LINE__);
        getch();
        }
    else
        {
        *inPtr = ptr;
        }
    }

void _operacion_incrementar_operadores(struct Operacion *mOp)
    {
    int t;
    mOp->maxOperadores=mOp->maxOperadores*2;
    t=mOp->maxOperadores;
    _expandir_memoria_int(&mOp->operador,t);
    _expandir_memoria_int(&mOp->operandoA,t);
    _expandir_memoria_int(&mOp->operandoB,t);
    _expandir_memoria_int(&mOp->operandoC,t);
    _expandir_memoria_int(&mOp->operandoD,t);
    _expandir_memoria_floatptr(&mOp->memorias,t);
    }


///@param mOp: Administrador de operaciones
///@param operador: Constante: por ejemplo SUMA, RGB_SUM, RGB_SAT, DIVISION, etc.
///@param OperandoA: El número de la entrada que se va a usar como operando.
#include <stdarg.h>
void operacion_agregar_op(struct Operacion *mOp, int operador, int operandoA, ... )
    {
    if(mOp->opAd >= mOp->maxOperadores)
        {
        printf("Expandiendo operadores %d->",mOp->opAd);
        _operacion_incrementar_operadores(mOp);
        printf("%d\r\n",mOp->maxOperadores);
        }
    int opAd = mOp->opAd;
    mOp->operador[opAd]=operador;
    mOp->operandoA[opAd]=operandoA;
//asignar memoria si es necesario
    switch(operador)
        {
        case ULTIMOS_10:
            mOp->memorias[opAd] = (float *) malloc (sizeof(float) * 10);
            break;
        }
//Carga operadores adicionales si es necesario
    va_list argumentos;
    switch(operador)
        {
        case RGB_SUM:  //ejemplo
            va_start(argumentos, operandoA);
            mOp->operandoB[opAd] = va_arg ( argumentos, int );
            mOp->operandoC[opAd] = va_arg ( argumentos, int );
            va_end(argumentos);
            break;
        default:
            break;
        }
    mOp->opAd++;
    }


void operacion_calcular(struct Operacion *mOp, float *entradas, int numEnt, float *salidas, int numSal)
    {
    int i,j;
    if(mOp->opAd != numSal)
        {
        printf("Error: mOp->opAd(%d) != numSal(%d)\r\n",mOp->opAd,numSal);
        getch();
        }
    //printf("salidas debe ser valido o esto truena\r\n");

    for(i=mOp->opAd-1; i>=0; i--)//operador especial
        {
        switch(mOp->operador[i])
            {
            case ANTERIOR:
                salidas[i]=salidas[i-1];
                break;
            case DELTA_ANTERIOR:
                {
                int A = mOp->operandoA[i];
                salidas[i]=entradas[A]-salidas[i-1];
                }
            break;
            case ULTIMOS_10:
                salidas[i]=mOp->memorias[i][0];
                mOp->memorias[i][9]=salidas[i-1];
                for(j=0; j<10-1; j++)
                    {
                    mOp->memorias[i][j]=mOp->memorias[i][j+1];
                    }
                break;
            }
        }

    for(i=0; i<mOp->opAd ; i++)
        {
        int A = mOp->operandoA[i];
        int B = mOp->operandoB[i];
        int C = mOp->operandoC[i];
        int D = mOp->operandoD[i];
        switch(mOp->operador[i])
            {
            case 0:  //(sin cambios)  //Tambien se puede usar para mover hacer historico de entradas
                salidas[i]=entradas[A];
                break;
            case CUADRADO:
                salidas[i]=entradas[A]*entradas[A];
                break;
            case CUBO:
                salidas[i]=entradas[A]*entradas[A]*entradas[A];
                break;
            case SENO:
                salidas[i]=sin(entradas[A]*3.1415926537*2);
                break;
            case COSENO:
                salidas[i]=cos(entradas[A]);
                break;
            case TANGENTE:
                salidas[i]=tan(entradas[A]);
                break;
            case SENO_H:
                salidas[i]=sinh(entradas[A]);
                break;
            case COSENO_H:
                salidas[i]=cosh(entradas[A]);
                break;
            case TANGENTE_H:
                salidas[i]=tanh(entradas[A]);
                break;
            case LOG:
                salidas[i]=log(entradas[A]);
                break;
            case LOG10:
                salidas[i]=log10(entradas[A]);
                break;
            case NOISE:
                salidas[i]=rand()/(float)(RAND_MAX/2.0)-1.0;  //de -1.0 a +1.0
                break;
            case TIME:
                break;
            /*case DELTA:
                salidas[i]=entradas[A]-salidas[i];
                break;
            case INCREMENTO:
                salidas[i]=entradas[A]+salidas[i];
                break;*/
//        case RESTA:
//            salidas[i]=entradas[A]-entradas[B];
//            break;
            case RGB_SUM:
                salidas[i]=entradas[A]+entradas[B]+entradas[C];
                break;
            }
        }
    return;
    }

void debug_op()
    {

    struct Operacion *mOp;
    printf("Creando operaciones\r\n");
    mOp = operacion_crear_operadores(2);
    printf("Agregando operaciones\r\n");
    operacion_agregar_op(mOp, IGUAL, 0);
    operacion_agregar_op(mOp, DELTA_ANTERIOR, 0);
    operacion_agregar_op(mOp, ANTERIOR, 0);
    operacion_agregar_op(mOp, ANTERIOR, 0);
    operacion_agregar_op(mOp, ANTERIOR, 0);
    operacion_agregar_op(mOp, ANTERIOR, 0);
    printf("Calculando\r\n");
#define OUT 6
    float Mentradas[1]= {0.1},Msalidas[OUT]= {0};
    operacion_calcular(mOp, Mentradas, 1, Msalidas,OUT);
    int i;
    for(i=0; i<OUT; i++)
        {
        printf("%f,\r\n",Msalidas[i]);
        }
    Mentradas[0]=0.8;
    operacion_calcular(mOp, Mentradas, 1, Msalidas,OUT);
    for(i=0; i<OUT; i++)
        printf("%f,\r\n",Msalidas[i]);
    Mentradas[0]=0.4;
    operacion_calcular(mOp, Mentradas, 1, Msalidas,OUT);
    for(i=0; i<OUT; i++)
        printf("%f,\r\n",Msalidas[i]);
    Mentradas[0]=0.2;
    operacion_calcular(mOp, Mentradas, 1, Msalidas,OUT);
    for(i=0; i<OUT; i++)
        printf("%f,\r\n",Msalidas[i]);
    }


//numEntradas debe coincidir con el tamano de la red sin caracteristicas u operadores
// *entradas son los datos del csv.
/*void fcalc_alimentar(struct *adm, float *entradas, int numEntradas)
{
adm.red.capa[0].entrada=entradas[forloop];
adm.red.capa[0].salida=f(adm.red.capa[0].entrada);
}
*/















///===================================================Administrador de red===============================

//otros nombres: RedControl RedIo

struct Administrador
    {
    int numEntradas;
    int numSalidas;
    int numEntradasAdicionales;
    struct Red *mRed;
    struct Operacion *mOp;
    float *Entradas;
    float *EntradasCalculadas;
    FILE *entrada, *entrenador, *salida;
    };

struct Administrador *administrador_crear(int numEntradas, int numSalidas)
    {
    struct Administrador *mAdm = (struct Administrador *) malloc(sizeof(struct Administrador));
    mAdm->numEntradas=numEntradas;
    mAdm->numSalidas=numSalidas;
    mAdm->mOp = operacion_crear_operadores(32);
    mAdm->entrada=NULL;
    mAdm->entrenador=NULL;
    mAdm->salida=NULL;
    return mAdm;
    }


void administrador_asignar_caracteristica_adicional(struct Administrador *mAdm, int operador, int cual_entrada)
    {
    operacion_agregar_op(mAdm->mOp, operador, cual_entrada);
    }




#include <stdarg.h>

//genera topologia final y crea la red
void administrador_generar_red(struct Administrador *mAdm, int num_capas_ocultas, ...)
    {
    int i;
    int topo[num_capas_ocultas+2];
    mAdm->numEntradasAdicionales = mAdm->mOp->opAd;
    topo[0]=mAdm->numEntradas+mAdm->numEntradasAdicionales;
    printf("Red creada con mAdm->numEntradas(%d)+numEntradasAdicionales(%d)\r\n",mAdm->numEntradas,mAdm->numEntradasAdicionales);
    va_list argumentos;
    va_start ( argumentos, num_capas_ocultas);
    for (i = 1; i<num_capas_ocultas+1; i++)
        {
        topo[i] = va_arg ( argumentos, int );
        }
    va_end ( argumentos );
    topo[i]=mAdm->numSalidas;
    mAdm->mRed = red_crear(topo, num_capas_ocultas+2);

    int numEntRed=mAdm->numEntradas+mAdm->numEntradasAdicionales;
    mAdm->Entradas = (float *) malloc(sizeof(float) * numEntRed);
    mAdm->EntradasCalculadas = &mAdm->Entradas[mAdm->numEntradas];
    }



//==================================================administrador de archivos============================================

//FILE *entrada=NULL, *entrenador=NULL, *salida=NULL;


void administrador_cargar_archivo_entrada(struct Administrador *mAdm, char *nombre)
    {
    if(mAdm->entrada == NULL)
        {
        mAdm->entrada = fopen(nombre, "r");
        }
    else
        {
        //rewind (mAdm->entrada);
        fclose(mAdm->entrada);
        mAdm->entrada = fopen(nombre, "r");
        }
    if(mAdm->entrada == NULL)
        {
        printf("Error de archivo de entrada");
        }
    else
        {
        //printf("entrada ok\r\n");
        }
    }


void administrador_cargar_archivo_entrenador(struct Administrador *mAdm, char *nombre)
    {
    if(mAdm->entrenador == NULL)
        {
        mAdm->entrenador = fopen(nombre, "r");
        }
    else
        {
        //rewind (mAdm->entrenador);
        fclose(mAdm->entrenador);
        mAdm->entrenador = fopen(nombre, "r");
        }

    if(mAdm->entrenador == NULL)
        {
        printf("Error de archivo de entrenador");
        }
    else
        {
        //printf("entrenador ok\r\n");
        }
    }


void administrador_cargar_archivo_salida(struct Administrador *mAdm, char *nombre)
    {
    if(mAdm->salida == NULL)
        {
        mAdm->salida = fopen(nombre, "wb+");
        }
    else
        {
        fclose(mAdm->salida);
        mAdm->salida = fopen(nombre, "wb+");
        }
    if(mAdm->salida == NULL)
        {
        printf("Error archivo de salida\r\n");
        }
    else
        {
        //printf("salida ok\r\n");
        }
    }


void administrador_alimentar_entrada(struct Administrador *mAdm)
    {
    int i;
    if(mAdm->entrada != NULL)
        {
//alimenta y calcula las caracteristicas de entrada
        for(i=0; i<mAdm->numEntradas-1; i++)
            {
            //fscanf(entrada, "%f",&mEntradas[i]);
            fscanf(mAdm->entrada, "%f,",&mAdm->Entradas[i]);
            }
        //fscanf(entrada, "%f\r\n",&mEntradas[i]);
        fscanf(mAdm->entrada, "%f\r\n",&mAdm->Entradas[i]);

        operacion_calcular(mAdm->mOp, mAdm->Entradas, mAdm->numEntradas, mAdm->EntradasCalculadas, mAdm->numEntradasAdicionales);
        red_alimentar(mAdm->mRed, mAdm->Entradas, mAdm->numEntradas+mAdm->numEntradasAdicionales);
        }
    else
        {
        printf("Nope");
        }

    }

void administrador_alimentar_entrenador(struct Administrador *mAdm)  //Conocido como "Aprender"
    {
    int i;
    float mSalidas[mAdm->numSalidas];
    if(mAdm->entrenador != NULL)
        {
        for(i=0; i<mAdm->numSalidas-1; i++)
            {
            fscanf(mAdm->entrenador, "%f,",&mSalidas[i]);
            }
        fscanf(mAdm->entrenador, "%f\r\n",&mSalidas[i]);
        red_aprender(mAdm->mRed, mSalidas, mAdm->numSalidas);
        }
    else
        {
        printf("Nope2");
        }
    }

void administrador_entrenar(struct Administrador *mAdm, int epoch, float valorDeParo)
    {
    int i;
    int j=0;
    for(i=0; i<epoch; i++)
        {
        administrador_cargar_archivo_entrada(mAdm,"in.csv");
        administrador_cargar_archivo_entrenador(mAdm,"res.csv");

        //llama a las 2 funciones que alimentan hasta acabar el archivo
        //repite el entrenamiento la cantidad de veces especificada:
        if(mAdm->entrada!=NULL && mAdm->entrenador != NULL)
            {
            while (!feof(mAdm->entrada) && !feof(mAdm->entrenador))
                {
                administrador_alimentar_entrada(mAdm);
                administrador_alimentar_entrenador(mAdm);
                j++;
                if(j==100000)
                    {
                    j=0;
                    printf("Epoch %d \tE=%f \tEp=%f\r\n",i,mAdm->mRed->error,mAdm->mRed->errorpromedioreciente);
                    }
                if(mAdm->mRed->errorpromedioreciente<valorDeParo)
                    {
                    //printf("Logrado");
                    //printf("E=%f \t\tEp=%f\r\n",mAdm->mRed->error,mAdm->mRed->errorpromedioreciente);
                    //return;  //DEBUG
                    }
                }
            }
        }

    }


///=================================================DEBUG section===================================================

//Luego Hay que hacer funciones que entrenen administradores de redes y seleccionen la mejor red o combinen resultados de administradores.

void neurona_debug()
    {
    struct Neurona *N = neurona_crear(4);
    neurona_print(N);
    }

void capa_debug()
    {
    struct Capa *mCapa1 = capa_crear(2, 2);
    struct Capa *mCapa2 = capa_crear(2, 3);
    struct Capa *mCapa3 = capa_crear(3, 1);
    capa_print(mCapa1);
    capa_print(mCapa2);
    capa_print(mCapa3);
    printf("Despues\r\n");

    capa_procesar_todo(mCapa1, mCapa2);
    capa_procesar_todo(mCapa2, mCapa3);
    capa_print(mCapa1);
    capa_print(mCapa2);
    capa_print(mCapa3);
    }

void red_debug()
    {
    srand (time(NULL));
#define IN_SIZE 2
#define OUT_SIZE 1
#define SAMPLES 4
    int topo[3]= {IN_SIZE,3,OUT_SIZE};
    struct Red *mRed = red_crear(topo, 3);

    float vin[SAMPLES][IN_SIZE]= {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
    float vout[SAMPLES][OUT_SIZE]= {{1.0},    {0.0},    {0.0},    {1.0}};

    red_alimentar(mRed, vin[3], IN_SIZE);
    red_resultados(mRed);
    printf("---------------------\r\n");
    int i,j;
    for(j=0; j<1000; j++)
        {
        for(i=0; i<SAMPLES; i++)
            {
            red_alimentar(mRed, vin[i], IN_SIZE);
            red_aprender(mRed, vout[i], OUT_SIZE);
            }
        }
    float test[2]= {0,1};
    red_alimentar(mRed, test, IN_SIZE);
    red_resultados(mRed);
    }

void red_debug2()
    {
    srand (time(NULL));
#define IN_SIZE 2
#define OUT_SIZE 1
#define SAMPLES 4
    int topo[3]= {IN_SIZE,3,OUT_SIZE};
    //struct Red *mRed = red_crear(topo, 3);
    struct Red *mRed=red_cargar("MiRed.txt");

    float vin[SAMPLES][IN_SIZE]= {{0.3,0.1},{0.8,0.2},{0.1,0.1},{0.2,0.6}};
    float vout[SAMPLES][OUT_SIZE]= {{0.4},    {1.0},    {0.2},    {0.8}};
    red_alimentar(mRed, vin[0], IN_SIZE);
    red_resultados(mRed);
    printf("---------------------\r\n");
    int i,j;
    for(j=0; j<10000; j++)
        {
        for(i=0; i<SAMPLES; i++)
            {
            red_alimentar(mRed, vin[i], IN_SIZE);
            red_aprender(mRed, vout[i], OUT_SIZE);
            }
        if(j%1==0)
            {
            printf("error:%f \t\tErrorprom:%f\r\n",mRed->error,mRed->errorpromedioreciente);
            }
        }
    float test[2]= {0.125,0.25};
    red_alimentar(mRed, test, IN_SIZE);
    red_resultados(mRed);
    //red_guardar(mRed, "MiRed.txt");
    }


//Diferenciar: red_crear_desde_archivo  y red_cargar_archivo
//red_crear_desde_archivo crea una nueva red desde un archivo
//red_cargar_archivo carga los pesos de la red desde un archivo, debe coincidir con la estructura de la red.
//
//administrador_crear_red_desde_archivo(): sin comentarios
//administrador_cargar_pesos_desde_archivo(): sin comentarios


void administrador_debug()
    {
    int i,j;
    srand (time(NULL));
    struct Administrador *mAdm = administrador_crear(1, 1);  //(int numEntradas, int numSalidas)
    printf("Creada\r\n");


    administrador_asignar_caracteristica_adicional(mAdm, IGUAL, 0);
    administrador_asignar_caracteristica_adicional(mAdm, DELTA_ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    /*administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
administrador_asignar_caracteristica_adicional(mAdm, IGUAL, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ANTERIOR, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);
    administrador_asignar_caracteristica_adicional(mAdm, ULTIMOS_10, 0);*/


    printf("Caracteristicas agregadas");


    administrador_generar_red(mAdm, 1,4);        //(int num_capas_ocultas, Numero de neuronas en capa)
    printf("red generada\r\n");
    red_resultados(mAdm->mRed);
    printf("Entrenamiento\r\n");

    for(j=0; j<1000; j++)
        {
        administrador_cargar_archivo_entrada(mAdm,"in.csv");
        administrador_cargar_archivo_entrenador(mAdm,"res.csv");
        administrador_entrenar(mAdm, 100, 0.0);
        char txt[140];
        sprintf(txt,"MiRed%d.txt",j);
        red_guardar(mAdm->mRed, txt);

        //red_cargar("MiRedA.txt");
        printf("Examen de evaluacion\r\n");
        administrador_cargar_archivo_entrada(mAdm,"in2.csv");
        sprintf(txt,"out%d.txt",j);
        administrador_cargar_archivo_salida(mAdm,txt);
        for(i=0; i<254000; i++)
            {
            administrador_alimentar_entrada(mAdm);
            //red_resultados(mAdm->mRed);
            fprintf(mAdm->salida,"%f,",mAdm->mRed->listaCapas[0]->listaNeuronas[0]->salida);
            fprintf(mAdm->salida,"%f\r\n",mAdm->mRed->listaCapas[mAdm->mRed->numCapas-1]->listaNeuronas[0]->salida);
            }
        }
    }

void debug_crear_datos()
    {
    FILE *entrada, *salida;
    srand (time(NULL));
    entrada = fopen("in.csv", "wb+");
    salida = fopen("res.csv", "wb+");
    int i;
    for(i=0; i<1000; i++)
        {
        float f=rand()/(float)RAND_MAX;
        fprintf(entrada,"%f\r\n",f);
        fprintf(salida,"%f\r\n",f/-2.0);
        }
    fclose(entrada);
    fclose(salida);
    }


//Por hacer:
//Funcion que borre neuronas
//Funcion que borre una capa de nauronas
//funcion que borre la red
//Funcion que clone una neurona (copia dura)
//funcion que clone una capa
//Funcion que clone la red (o los datos de la red) para guardar el mejor resultado

//Talves renombrar administrador a RedIO para luego crear un administrador que gestione redes
//Funcion que reciba una sola entrada como argumento, ejemplo ULTIMOS_10_DATOS, ULTIMOS_100_DATOS y pueda crear 10 neuronas que almacenen 10 datos historicos.
//Agregar derivada y ULTIMAS_10_DERIVADAS como caracteristicas.
//Agregar integral.


//

///==============================================Main=======================================================

int main()
    {
    //debug_op();
    administrador_debug();
    //debug_crear_datos();
    return 0;
    }





///Savefile.c
///==================================================Binary file recording library=========================================================

#include <stdarg.h>
#include <time.h>
int archivo_fprintf(const char *format, ...);
int archivo_fscanf(const char *format, ...);
void archivo_nuevo();
void archivo_cerrar();
FILE *archivo;
unsigned char filename[100];

int archivo_fprintf(const char *format, ...)
    {
    va_list p;
    int n;
    va_start(p, format);
    vfprintf(stdout, format, p);
    n = vfprintf(archivo, format, p);
    va_end(p);
    return n;
    }

int archivo_fscanf(const char *format, ...)
    {
    va_list p;
    int n;
    va_start(p, format);
    n = vfscanf(archivo, format, p);
    va_end(p);
    return n;
    }

void archivo_nuevo(char *filename)
    {
    archivo=fopen((const char *)filename,"wb+");
    if(archivo==NULL)
        {
        printf("Error al crear archivo\r\n");
        exit(3);
        }
    printf("%s\r\n",filename);
    }

void archivo_abrir(char *filename)
    {
    archivo=fopen((const char *)filename,"rb+");
    if(archivo==NULL)
        {
        printf("Error al crear archivo\r\n");
        exit(3);
        }
    printf("%s\r\n",filename);
    }


void archivo_cerrar()
    {
    if(fclose(archivo)!=0)
        {
        printf("Error al cerrar archivo archivo_cerrar()\r\n");
        }
    archivo=NULL;
    }

