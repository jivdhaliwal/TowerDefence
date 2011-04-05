/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.awt.Font;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Color;
import org.newdawn.slick.TrueTypeFont;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.MouseListener;
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
    int mouseCounter;
    // Map used to find critter paths
    private PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private Path path;
    private TiledMap map;
    private Image pathSprite;
    private Image towerSprite;

    CritterManager critterWave;
    ArrayList<Critter> critterList;
    ArrayList<Tower> towerList = new ArrayList<Tower>();

    Critter critter = null;
    ParticleSystem ps;
    private TrueTypeFont trueTypeFont;
    private Tower testTower;
    private Tower testTower2;
    private TowerManager towerFactory;


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

        towerFactory = new TowerManager();


        critterCount = 20000;
        critterWave = new CritterManager(
                new Vector2f((float) (32 * (map.getWidth() - 1)), (float) (32 * (map.getHeight() - 1))),
                finder,critterCount);

        critterList = critterWave.getCritters();

        Font font = new Font("Verdana", Font.PLAIN, 20);
        trueTypeFont = new TrueTypeFont(font, true);

    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        map.render(0, 0);

        // Render the path for testing
        /*
        for (int x = 0; x < map.getWidth(); x++) {
            for (int y = 0; y < map.getHeight(); y++) {
                if (path != null) {
                    if (path.contains(x, y)) {
                        pathSprite.draw(x * 32, y * 32);
                    }
                }
            }
        }
         * 
         */

        critterWave.render(container, game, g);

        towerFactory.render(container, game, g);
        
        trueTypeFont.drawString((map.getWidth() * 32) - 200, 50, "# of Critters : " + String.valueOf(critterWave.getCritters().size()), Color.white);
        trueTypeFont.drawString((map.getWidth() * 32) - 200, 100, "# of Towers : " + String.valueOf(towerFactory.getTowers().size()), Color.white);


    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        Input input = container.getInput();

        if (input.isKeyPressed(Input.KEY_F2)) {
            game.enterState(TowerDefence.PATHTESTSTATE);
        }

        mouseCounter -= delta;

        critterWave.update(container, game, delta);

        towerFactory.updateCritterList(critterWave.getCritters());
        towerFactory.update(container, game, delta);

        Input i = container.getInput();i.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
                if (mouseCounter <= 0) {
                    int currentXTile = (int) Math.floor((x / 32));
                    int currentYTile = (int) Math.floor((y / 32));
                    try {
                        towerFactory.addTower(String.valueOf(x), new Vector2f(currentXTile * 32, currentYTile * 32));
                        mouseCounter=50;
                    } catch (SlickException ex) {
                        Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }

            }

            public void mousePressed(int button, int x, int y) {
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
            }

            public void mouseDragged(int oldx, int oldy, int newx, int newy) {
            }

            public void setInput(Input input) {
            }

            public boolean isAcceptingInput() {
                return true;
            }

            public void inputEnded() {
            }

            public void inputStarted() {
            }
        });

    }
}
