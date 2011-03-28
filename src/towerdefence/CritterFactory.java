package towerdefence;

import java.util.ArrayList;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.util.pathfinding.Path;
import org.newdawn.slick.util.pathfinding.PathFinder;
import towerdefence.engine.component.CritterFollowPathComponent;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.component.TopDownMovement;
import towerdefence.engine.entity.Entity;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */



public class CritterFactory {

    int numCritters;

    Vector2f initialPos;
    Image testerSprite;
    PathFinder finder;
    Path path;

    ArrayList<Entity> critterList = new ArrayList<Entity>();

    public CritterFactory(Vector2f initialPos, PathFinder finder) throws SlickException {
        //this.numCritters = numCritters;
        this.initialPos = initialPos;
        this.finder = finder;
        this.path = finder.findPath(new UnitMover(3), getTilePosition(initialPos.x,32),
                getTilePosition(initialPos.y,32), 1,1);
        testerSprite = new Image("data/sprites/positionTester.png");
    }

    /*
     * Add's critter to ArrayList of critters
     */
    public void addCritter(String id) {
        Entity critter = new Entity(id);
        critter.setPosition(initialPos);

        critter.AddComponent(new ImageRenderComponent("CritterRender", testerSprite));
        critter.AddComponent(new TopDownMovement("CritterMovement"));
        critter.AddComponent(new CritterFollowPathComponent("CritterPath", finder, path));

        critterList.add(critter);
    }

    public void removeCritter(Entity e) {
        critterList.remove(e.getId());
    }

    public void generateCritters(int numCritters) {
        for(int i=0;i<numCritters;i++) {
            addCritter(String.valueOf(i));
        }
    }

    public ArrayList<Entity> getCritters() {
        return critterList;
    }

    public int getTilePosition(float position, int tilesize)
    {
        return ((int) (Math.floor(position / tilesize)));
    }

}
