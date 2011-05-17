package towerdefence.engine.levelLoader;


import org.newdawn.slick.SlickException;
import org.newdawn.slick.util.xml.SlickXMLException;
import org.newdawn.slick.util.xml.XMLElement;
import org.newdawn.slick.util.xml.XMLElementList;
import org.newdawn.slick.util.xml.XMLParser;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class LevelLoader {

    private Wave[] waveList;
    
    private static XMLElement root;
    
    private String mapPath;
    
    private final String levelName;

    public LevelLoader(String filepath) throws SlickException {
        XMLParser parser = new XMLParser();

        root = parser.parse(filepath);
        
        levelName = root.getAttribute("name");
        
        waveList = new Wave[root.getChildrenByName("Waves").get(0).getChildren().size()];
        
        mapPath = root.getChildrenByName("Tilemap").get(0).getAttribute("path");

        loadWaves();
    }


    private void loadWaves() throws SlickXMLException {
        int critterType;
        int numCritters;
        int timeToWait;

        XMLElementList waves = root.getChildrenByName("Waves").get(0).getChildren();
        for(int i=0;i<waves.size();i++){
            critterType = waves.get(i).getIntAttribute("type");
            numCritters = waves.get(i).getIntAttribute("count");
            timeToWait = waves.get(i).getIntAttribute("waitingTime");
            Wave wave = new Wave(critterType, numCritters, timeToWait);
            waveList[i] = wave;
        }
        
    }

    public String getMapPath() {
        return mapPath;
    }

    public Wave getWave(int index) {
        return waveList[index];
    }

    public int getNumWaves() {
        return waveList.length;
    }

    /**
     * @return the levelName
     */
    public String getLevelName() {
        return levelName;
    }
}