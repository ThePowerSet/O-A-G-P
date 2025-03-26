Ecco un riassunto delle principali azioni che hai intrapreso e dei progressi fatti finora:

Problemi riscontrati:
	1.	Problema di compatibilità con la GPU e TensorFlow:
	•	Hai riscontrato un problema con la versione di TensorFlow e la compatibilità con la GPU del tuo Mac M1, e inizialmente avevi problemi a far funzionare correttamente l’accelerazione hardware.
	2.	Errore nei test (related to imports e dipendenze):
	•	C’erano conflitti di versione tra TensorFlow e alcune librerie, oltre a problemi con le dipendenze per far funzionare correttamente il codice nella repository.

Cosa è stato fatto:
	1.	Installazione dei pacchetti necessari:
	•	Hai installato tensorflow-macos e tensorflow-metal per abilitare il supporto per la GPU sul Mac M1, visto che TensorFlow non distribuisce pacchetti precompilati per l’architettura ARM64 di Apple.
	•	Hai anche risolto i conflitti di versioni tra TensorFlow e alcune librerie come keras, tensorboard, ecc.
	2.	Esecuzione dei test:
	•	Dopo aver risolto i conflitti di dipendenza, sei riuscito a eseguire correttamente i test, e tutti i test sono passati, a parte alcuni avvisi di deprecazione che non influiscono direttamente sul funzionamento del codice.
	3.	Uso di pytest:
	•	Hai eseguito pytest con successo, ottenendo risultati positivi per la maggior parte dei test.

Stato attuale:
	1.	Ambiente funzionante: Ora hai un ambiente Python funzionante con TensorFlow correttamente configurato per il tuo Mac M1.
	2.	Test passati: I test della repository sono passati, il che indica che il codice è in buona parte funzionante.
	3.	Problemi di esecuzione degli script: L’esecuzione degli script dalla cartella examples/uci/ non è andata a buon fine a causa di un possibile errore nel percorso dei file. Probabilmente un errore di percorso o di disponibilità dei file nella cartella.

Quando riprenderai il progetto:
	1.	Controlla la struttura delle cartelle:
	•	Verifica che i file necessari siano presenti, in particolare quelli all’interno della cartella examples/uci/. Usa ls per esplorare le cartelle e assicurarti che i file esistano.
	2.	Esegui gli script da directory corretta:
	•	Assicurati di essere nella directory giusta prima di eseguire i comandi. Puoi navigare usando il comando cd per spostarti nelle cartelle appropriate.
	3.	Riprendi i test:
	•	I test sono passati, quindi puoi riprendere da dove avevi lasciato e continuare a esplorare o modificare il codice.
	4.	Risoluzione di eventuali altri problemi:
	•	Se dovessi riscontrare altri problemi, puoi provare a fare debug degli script specifici o aggiornare ulteriormente le dipendenze per gestire possibili conflitti.

In sintesi:

Sei riuscito a risolvere i principali problemi di compatibilità con TensorFlow su Mac M1, hai eseguito i test con successo e hai avviato la configurazione del progetto. La situazione da cui ripartirai quando riaprirai il progetto sarà un ambiente di lavoro stabile con i test passati, ma dovrai verificare la corretta esecuzione degli script nella cartella examples/uci/.

Se hai bisogno di ulteriori chiarimenti o aiuto con specifici passaggi, fammi sapere!



Run di example_autompg :
mi ha dato alcuni errori tra cui che mancava tensorboard, l'ho installato manualmente 

problema quando runno pytest dalla cartella test, mentre quando sono nella cartella generale me lo runna, 
il problema credo fosse che non mi riconosceva la vesione di python e stava su una 3.12.

mi capita che esce dall'environment anche se mi dice che ci sto ancora dentro e anche se ri setto l'interprete, devo killare il terminale
aprirne un altro e fare deactivate


## this is a message i obtain when using gpflow but it's not critical so until i don't have problems i won't fix this,
## it basically tells that tensorflow can't use the gpu for real, even though it recognizes it
>>> kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale)
2025-03-25 15:35:39.492810: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-25 15:35:39.493970: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)


# this is a warning message, it's weird because i have tensorboard, however if i do control c a couple times it goes
# I got into some problems also importing gpflow, i have to import before tensorglow than gpflow
>>> import tensorflow as tf
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.

