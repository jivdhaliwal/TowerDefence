package towerdefence;

import java.util.ArrayList;
import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.PathFinder;
import towerdefence.engine.Player;
import towerdefence.engine.ResourceManager;
import towerdefence.engine.component.CritterAnimationComponent;
import towerdefence.engine.component.CritterFollowPathComponent;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */



public class CritterManager {

    int generateCounter;

    Vector2f initialPos;
    PathFinder finder;
    Path path;

    private ArrayList<Critter> critterList = new ArrayList<Critter>();
    private Animation[][] critterAnimation = new Animation[3][];
    
    public CritterManager(int startX, int startY, int goalX, int goalY,
            PathFinder finder) throws SlickException {
        this.finder = finder;
        this.path = finder.findPath(new UnitMover(3), startX, startY, goalX, goalY);
        this.initialPos = new Vector2f(startX * GameplayState.TILESIZE, startY * GameplayState.TILESIZE);
        
        loadAnimations();

    }


    /*
     * Add's critter to ArrayList of critters
     */
    public void addCritter(String id, int critterType) throws SlickException {
        Critter critter = new Critter(id);
        critter.setPosition(initialPos);
        critter.setType(critterType);

        critter.AddComponent(new CritterAnimationComponent(id,critterAnimation[critterType]));
        critter.AddComponent(new CritterFollowPathComponent("CritterPath", path, critterType));
        critterList.add(critter);
    }

    public ArrayList<Critter> getCritters() {
        return critterList;
    }

    public int getTilePosition(float position, int tilesize)
    {
        return ((int) (Math.floor(position / tilesize)));
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        generateCounter-=delta;

        for (Critter enemy : critterList) {
            enemy.update(container, game, delta);
        }

        // Remove dead critters
        for (int i = 0; i < critterList.size(); i++) {
            if (critterList.get(i).isDelete()) {
                // -1 = critter reached goal
                critterList.remove(i);
            } else if (critterList.get(i).isDead()) {
                Player.getInstance().killCritter(critterList.get(i).getType());
                critterList.remove(i);
            }
        }

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {

        for (Critter enemy : critterList) {
            enemy.render(container, game, g);
        }

    }
    
    public void loadAnimations() {

    	Animation[] normalAnimation = {ResourceManager.getInstance().getAnimationFromPack("NORMAL_CRITTER_UP"),
    			ResourceManager.getInstance().getAnimationFromPack("NORMAL_CRITTER_DOWN"),
    			ResourceManager.getInstance().getAnimationFromPack("NORMAL_CRITTER_LEFT"),
    			ResourceManager.getInstance().getAnimationFromPack("NORMAL_CRITTER_RIGHT")};
    		critterAnimation[Critter.NORMAL] = normalAnimation;
    		
		Animation[] fireAnimation = {ResourceManager.getInstance().getAnimationFromPack("FIRE_CRITTER_UP"),
				ResourceManager.getInstance().getAnimationFromPack("FIRE_CRITTER_DOWN"),
				ResourceManager.getInstance().getAnimationFromPack("FIRE_CRITTER_LEFT"),
				ResourceManager.getInstance().getAnimationFromPack("FIRE_CRITTER_RIGHT")};
			critterAnimation[Critter.FIRE] = fireAnimation;
			
		Animation[] iceAnimation = {ResourceManager.getInstance().getAnimationFromPack("ICE_CRITTER_UP"),
				ResourceManager.getInstance().getAnimationFromPack("ICE_CRITTER_DOWN"),
				ResourceManager.getInstance().getAnimationFromPack("ICE_CRITTER_LEFT"),
				ResourceManager.getInstance().getAnimationFromPack("ICE_CRITTER_RIGHT")};
			critterAnimation[Critter.ICE] = iceAnimation;
    }

}
