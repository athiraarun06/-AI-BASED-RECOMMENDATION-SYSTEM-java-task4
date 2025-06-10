import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class ProductRecommender {
    public static void main(String[] args) {
        try {
            DataModel model = new FileDataModel(new File("data/user_preferences.csv"));

            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            LongPrimitiveIterator users = model.getUserIDs();

            while (users.hasNext()) {
                long userId = users.nextLong();
                List<RecommendedItem> recommendations = recommender.recommend(userId, 2);
                System.out.println("User " + userId + " recommendations:");
                for (RecommendedItem item : recommendations) {
                    System.out.println("  Item: " + item.getItemID() + " | Preference: " + item.getValue());
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
