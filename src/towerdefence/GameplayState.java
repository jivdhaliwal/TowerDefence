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
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Tower;

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
    private Image towerSprite;

    CritterFactory critterFactory;
    ArrayList<Critter> critters;
    Critter critter = null;
    ParticleSystem ps;
    private TrueTypeFont trueTypeFont;
    private Tower testTower;


    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {


        pathSprite = new Image("data/sprites/path.png");
        towerSprite = new Image("data/sprites/towers/firetower.png");
        map = new TiledMap("data/maps/path1_3.tmx");

        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        path = finder.findPath(new UnitMover(3), map.getWidth() - 1, map.getHeight() - 1, 1, 1);


        critterCount = 200;

        critterFactory = new CritterFactory(
                new Vector2f((float) (32 * (map.getWidth() - 1)), (float) (32 * (map.getHeight() - 1))),
                finder);

        critters = critterFactory.getCritters();

        testTower = new Tower("TestTower");
        testTower.setPosition(new Vector2f(10*32,9*32));
        testTower.AddComponent(new ImageRenderComponent("CritterRender", towerSprite));

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

        for (Critter enemy : critters) {
            enemy.render(container, game, g);
        }

        testTower.render(container, game, g);

        trueTypeFont.drawString((map.getWidth() * 32) - 200, 50, "# of Critters : " + String.valueOf(critters.size()), Color.white);

    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        Input input = container.getInput();

        if (input.isKeyPressed(Input.KEY_F2)) {
            game.enterState(TowerDefence.PATHTESTSTATE);
        }

        generateCounter -= delta;
//        critters = critterFactory.getCritters();

        for (Critter enemy : critters) {
            enemy.update(container, game, delta);
            
        }

        for(int i=0;i<critters.size();i++) {
            if(critters.get(i).isDead()) {
                critters.remove(i);
            }
        }

        testTower.update(container, game, delta);
        testTower.updateCritterList(critters);

        if (generateCounter < 0) {
            if (critterCount > 0) {

                critterFactory.addCritter(String.valueOf(critterCount + 100));
                critterCount--;
            }
            generateCounter = 500;
        }

    }
}
