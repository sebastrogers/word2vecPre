import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import ch.qos.logback.core.read.ListAppender;

public class LendoModelo {

	public static void main(String[] args) throws IOException {

		//LENDO MODELO TREINADO
		Word2Vec vec = WordVectorSerializer.loadFullModel("pathToSaveModelPreconceitoNOVO5.txt");
		//Word2Vec vec = WordVectorSerializer.readWord2Vec(new File("pathToSaveModelPreconceito.txt"));

		Collection<String> lst = vec.wordsNearest("negro", 10);
		System.out.println(lst);       
		
		ArrayList<String>  p = Somatorio.ler("ListaP.txt");
		ArrayList<String>  sp = Somatorio.ler("ListaSP.txt");
		
		double cosSim = 0;
		for (String word : p) {

			cosSim = vec.similarity("negro", word);
			
			System.out.println("Palavra + sentimento negativo");
			System.out.println("negro + " + word + ": " + cosSim);
			System.out.println("-----------------------------------------");
			
		}
		
		for (String word : sp) {
			
			System.out.println("Palavra + sentimento positivo");
			cosSim = vec.similarity("negro", word);
			System.out.println("negro + " + word + ": " + cosSim);
			System.out.println("-----------------------------------------");	
			
		}

	}

}