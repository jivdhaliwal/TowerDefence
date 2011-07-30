package towerdefence.engine;

import org.newdawn.slick.Image;
import org.newdawn.slick.Music;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.Sound;

/**
 *
 * Singleton for loading game resources
 * 
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class ResourceManager {
    
    private static ResourceManager resourceLoader = null;
    
    // GUI Resources
    
    private ResourceManager() throws SlickException {
        loadResources();
    }
    
    public static ResourceManager getInstance() throws SlickException {
        if(resourceLoader==null) {
            resourceLoader = new ResourceManager();
        }
        return resourceLoader;
    }
    
    private void loadResources() throws SlickException {
        
        
    	
    }

}
