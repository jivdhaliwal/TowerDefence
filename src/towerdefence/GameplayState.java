/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.awt.Font;
import org.newdawn.slick.Color;
import org.newdawn.slick.TrueTypeFont;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.particles.ParticleSystem;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;

import towerdefence.engine.entity.Entity;
import towerdefence.engine.pathfinding.PathMap;
import towerdefence.engine.pathfinding.UnitMover;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;
    int critterCount;
    int generateCounter;
    // Map used to find critter paths
    private PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private Path path;
    private TiledMap map;
    private Image pathSprite;
    CritterFactory critterFactory;
    ArrayList<Entity> critters = new ArrayList<Entity>();
    Entity critter = null;
    ParticleSystem ps;
    private TrueTypeFont trueTypeFont;

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {


        pathSprite = new Image("data/sprites/path.png");
        map = new TiledMap("data/maps/path1_3.tmx");

        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        path = finder.findPath(new UnitMover(3), map.getWidth() - 1, map.getHeight() - 1, 1, 1);


        critterCount = 1000;

        critterFactory = new CritterFactory(
                new Vector2f((float) (32 * (map.getWidth() - 1)), (float) (32 * (map.getHeight() - 1))),
                finder);

        critters = critterFactory.getCritters();

        Font font = new Font("Verdana", Font.PLAIN, 20);
        trueTypeFont = new TrueTypeFont(font, true);


    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        map.render(0, 0);

        // Render the path for testing
        for (int x = 0; x < map.getWidth(); x++) {
            for (int y = 0; y < map.getHeight(); y++) {
                if (path != null) {
                    if (path.contains(x, y)) {
                        pathSprite.draw(x * 32, y * 32);
                    }
                }
            }
        }

        for (Entity enemy : critters) {
            enemy.render(container, game, g);
        }

        trueTypeFont.drawString((map.getWidth() * 32) - 200, 50, "# of Critters : " + String.valueOf(critters.size()), Color.white);

    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        Input input = container.getInput();

        if (input.isKeyPressed(Input.KEY_F2)) {
            game.enterState(TowerDefence.PATHTESTSTATE);
        }

        generateCounter -= delta;
        critters = critterFactory.getCritters();

        for (Entity enemy : critters) {
            enemy.update(container, game, delta);
        }

        if (generateCounter < 0) {
            if (critterCount > 0) {

                critterFactory.addCritter(String.valueOf(critterCount + 100));
                critterCount--;
            }
            generateCounter = 0;
        }

    }
}
