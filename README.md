Kodovi su napisani u sklopu predmeta Seminar i Projekt cija je tema CUDA implementacija BFS algoritma.
Primjer na kojem svi kodovi rade dostupan je u datoteci celegansneural.txt.

Rezultati se nalaze u datoteci rezultati.pdf.

1. Sekvencijska implementacija BFS-a (bfs_seq.cpp)
  - kod se kompajlira koristeći g++ kompajler
      >> g++ bfs_seq.cpp -o bfs_seq
  - parametri se unose preko standardnog ulaza
      >> bfs_seq < input_file.txt
  - izlaz se ispisuje u datoteku results_seq.txt

2. Paralelna CUDA implementacija BFS-a (bfs_par.cu, bfs_par_2.cu)
  - kod se kompajlira koristeći nvcc kompajler
  - pri kompajliranju bfs_par_2.cpp potrebno je dodati zastavicu -arch=sm_20
      >> nvcc bfs_par_2.cpp -o -arch=sm_20 bfs_par_2
  - parametri se unose preko standardnog ulaza (bfs_par < input_file.txt)
  - izlaz se ispisuje u datoteku results_par.txt, odnosno results_par_2.txt

3. Pokušaj unaprijeđenja paralelne CUDA implementacije BFS-a
  - nvcc bfs_novi.cu -arch=sm_20 -o bfs_par_3
  - ulazna datoteka unosi se kao argument iz komandne linije
  - izlaz se ispisuje u datoteku results_par.txt
